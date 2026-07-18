"""语义合成器 — 用 judge provider 分析多模型回答，生成结构化合成报告。

设计要点：
- 组装所有 provider 的多轮回答为结构化文本
- 调用 judge provider（LLM）生成 JSON 合成报告
- JSON 解析失败 → retry with explicit instruction
- retry 失败 → fallback 手动构建 SynthesisReport
"""

import json
import re
from typing import Optional

from .protocol import (
    DeliberationRound,
    ProviderResponse,
    SynthesisConsensus,
    SynthesisDivergence,
    SynthesisInsight,
    SynthesisReport,
    SynthesisStatus,
)
from .providers.base import BaseProvider


# ── Prompt 模板 ────────────────────────────────────────────────────

SYNTHESIS_PROMPT_TEMPLATE = """你是一个审议裁判。请分析以下多个AI模型对同一问题的回答，生成结构化合成报告。

## 原始问题
{prompt}

## 各模型回答
{answers}

## 请输出JSON格式的合成报告
{{
  "executive_summary": "一句话总结各方结论",
  "consensus": [
    {{"point": "共识点", "agreed_by": ["model1", "model2"], "confidence": "high"}}
  ],
  "divergences": [
    {{"point": "分歧点", "positions": {{"model1": "立场A", "model2": "立场B"}}, "critical": false}}
  ],
  "insights": [
    {{"point": "独特洞察", "provider_name": "model1", "category": "技术方案"}}
  ]
}}

只输出JSON，不要其他文字。"""

RETRY_PROMPT_TEMPLATE = """你上次的输出不是有效的JSON。请严格按照以下JSON格式重新输出合成报告。

## 原始问题
{prompt}

## 各模型回答
{answers}

## 必须输出以下格式的纯JSON（不要markdown代码块，不要额外文字）：
{{
  "executive_summary": "一句话总结",
  "consensus": [{{"point": "...", "agreed_by": ["name1"], "confidence": "high|medium|low"}}],
  "divergences": [{{"point": "...", "positions": {{"name1": "立场", "name2": "立场"}}, "critical": true/false}}],
  "insights": [{{"point": "...", "provider_name": "name", "category": "分类"}}]
}}

只输出JSON："""

# ── Synthesizer ────────────────────────────────────────────────────

class Synthesizer:
    """语义合成器 — 用 judge provider 分析多模型回答。"""

    MAX_RETRIES = 2

    def __init__(self, judge_provider: Optional[BaseProvider] = None):
        """
        Args:
            judge_provider: 用于合成的 judge provider（如 claude、gpt-5.6-sol），可为None
        """
        self.judge = judge_provider

    def synthesize(
        self,
        prompt: str,
        rounds: list[DeliberationRound],
    ) -> SynthesisReport:
        """对多轮审议结果进行语义合成。

        Args:
            prompt: 原始用户问题
            rounds: 所有审议轮次

        Returns:
            SynthesisReport 结构化合成报告
        """
        # 1. 组装所有回答文本
        answers_text = self._build_answers_text(rounds)
        if not answers_text.strip():
            return self._build_empty_report("所有轮次均无有效回答")

        # 2. 首次尝试
        raw = self._call_judge(SYNTHESIS_PROMPT_TEMPLATE.format(
            prompt=prompt,
            answers=answers_text,
        ))

        parsed = self._parse_json(raw)
        if parsed is not None:
            return self._build_report(parsed, status=SynthesisStatus.SUCCESS)

        # 3. retry 循环
        for attempt in range(self.MAX_RETRIES):
            retry_prompt = RETRY_PROMPT_TEMPLATE.format(
                prompt=prompt,
                answers=answers_text,
            )
            raw = self._call_judge(retry_prompt)
            parsed = self._parse_json(raw)
            if parsed is not None:
                return self._build_report(parsed, status=SynthesisStatus.SUCCESS)

        # 4. fallback
        return self._build_fallback(prompt, rounds)

    # ── 内部方法 ─────────────────────────────────────────────────

    def _call_judge(self, synthesis_prompt: str) -> str:
        """调用 judge provider，返回原始文本。"""
        resp = self.judge.invoke(synthesis_prompt)
        if resp.error:
            raise RuntimeError(f"Judge provider 调用失败: {resp.error}")
        return resp.text

    def _build_answers_text(self, rounds: list[DeliberationRound]) -> str:
        """将所有轮次的所有回答组装为结构化文本。"""
        parts = []
        for rnd in rounds:
            parts.append(f"\n### 第 {rnd.round_num} 轮 ({rnd.round_type.value})")
            for resp in rnd.responses:
                if resp.ok:
                    parts.append(
                        f"\n**{resp.provider_name}** ({resp.model}):\n{resp.text}\n"
                    )
                else:
                    parts.append(
                        f"\n**{resp.provider_name}** ({resp.model}): "
                        f"[错误: {resp.error}]\n"
                    )
        return "\n".join(parts)

    def _parse_json(self, text: str) -> Optional[dict]:
        """从文本中提取 JSON 对象。

        处理常见情况：
        - 纯 JSON
        - ```json ... ``` 代码块
        - ``` ... ``` 代码块
        - 首尾有额外文字
        """
        if not text:
            return None

        # 尝试 1: 直接解析
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # 尝试 2: 提取 ```json 代码块
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 尝试 3: 查找第一个 { 和最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        # 尝试 4: 修复常见 JSON 错误（未转义换行符、尾部逗号等）
        try:
            cleaned = re.sub(r",\s*}", "}", text[start:end + 1])
            cleaned = re.sub(r",\s*]", "]", cleaned)
            return json.loads(cleaned)
        except (json.JSONDecodeError, UnboundLocalError):
            pass

        return None

    def _build_report(self, data: dict, status: SynthesisStatus) -> SynthesisReport:
        """从解析后的字典构建 SynthesisReport。"""
        consensus = [
            SynthesisConsensus(
                point=c.get("point", ""),
                agreed_by=c.get("agreed_by", []),
                confidence=c.get("confidence", "medium"),
            )
            for c in data.get("consensus", [])
        ]
        divergences = [
            SynthesisDivergence(
                point=d.get("point", ""),
                positions=d.get("positions", {}),
                critical=d.get("critical", False),
            )
            for d in data.get("divergences", [])
        ]
        insights = [
            SynthesisInsight(
                point=i.get("point", ""),
                provider_name=i.get("provider_name", ""),
                category=i.get("category", ""),
            )
            for i in data.get("insights", [])
        ]
        return SynthesisReport(
            executive_summary=data.get("executive_summary", ""),
            consensus=consensus,
            divergences=divergences,
            insights=insights,
            judge_model=self.judge.model,
            status=status,
        )

    def _build_empty_report(self, reason: str) -> SynthesisReport:
        """所有回答均为空时的空报告。"""
        return SynthesisReport(
            executive_summary=f"无法生成合成报告: {reason}",
            judge_model=self.judge.model,
            status=SynthesisStatus.EMPTY,
            error=reason,
        )

    def _build_fallback(
        self,
        prompt: str,
        rounds: list[DeliberationRound],
    ) -> SynthesisReport:
        """JSON 解析全部失败时的降级方案 — 手动构建简单 SynthesisReport。

        从所有回答中提取 provider 名称作为基本元数据，
        不做语义分析。
        """
        all_ok = [
            r for rnd in rounds for r in rnd.responses if r.ok
        ]
        provider_names = list({r.provider_name for r in all_ok})

        # 收集所有回答摘要（前200字符）
        summaries = []
        for r in all_ok:
            preview = r.text[:200].replace("\n", " ").strip()
            summaries.append(f"- **{r.provider_name}** ({r.model}): {preview}...")

        executive_summary = (
            f"自动降级合成（JSON 解析失败）。共 {len(all_ok)} 个有效回答，"
            f"来自 {len(provider_names)} 个模型：{', '.join(provider_names)}。"
        )

        return SynthesisReport(
            executive_summary=executive_summary,
            consensus=[
                SynthesisConsensus(
                    point=f"各模型均对问题「{prompt[:80]}...」给出了回答",
                    agreed_by=provider_names,
                    confidence="low",
                )
            ],
            divergences=[],
            insights=[
                SynthesisInsight(
                    point=f"来自 {r.provider_name} 的回答预览: {r.text[:200]}...",
                    provider_name=r.provider_name,
                    category="降级合成",
                )
                for r in all_ok[:5]  # 最多 5 条
            ],
            judge_model=self.judge.model,
            status=SynthesisStatus.FALLBACK,
            error="JSON 解析失败，使用降级合成",
        )
