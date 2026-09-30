"""Build self-contained consultant prompts from the current screenplay + shot list.

    python3 film/consult/make_bundles.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILM = HERE.parent

GEMINI = """你是动画短片《光年》LIGHT-YEARS 的**科学顾问**。这部约 4 分半的短片准备投戛纳短片单元，全片用代码渲染成点彩风格，无对白，只有中英双语字卡。下面附上完整剧本和分镜表。

请从科学准确性和可信度的角度审读，重点检查：
1. 天体物理：银晕中年老红巨星周围的文明；光走七万三千光年；向所有方向膨胀的光壳；光壳扫过星云时的照亮与"光回波"；银河结构；视线从银面上方越过银心。
2. 人眼与视觉：暗适应需要多长时间、看到的星空是什么样子；单个光子能否被察觉（Tinsley et al. 2016）；暗视觉下的颜色感知（例如浦肯野效应、视杆细胞无色觉），这会影响那颗"金色的星"。
3. 考古：布隆伯斯洞穴约 7.3 万年前的交叉线绘画（九条线、赭石笔、硅质岩石片），以及约 7.5 万年前的织纹螺贝壳串珠。
4. 光污染：Falchi et al. 2016。
5. 日常物理：700 米外闪光与声音的延迟、城市停电的扩散方式、手电光柱在大气中如何可见、离开大气层后为什么看不见。

请输出（中文，900 字以内）：
- **问题清单**：每条写明哪里不对或不可信、严重程度（高/中/低），以及一个**保留诗意意图**的修改建议。
- **3–5 个能让画面更可信的准确细节**：例如红巨星表面的样子、暗适应后的星空、布隆伯斯石片的真实比例等。

电影诗意允许存在，但请指出那些懂行的观众会出戏的地方。
"""

GROK = """你是动画短片《光年》LIGHT-YEARS 的**剧本顾问**，请按一位严苛的戛纳短片评委的标准审读。这部约 4 分半的短片全片用代码渲染成点彩风格（一切由光点构成，人物多为剪影、轮廓光、手部和眼睛特写），无对白，只有中英双语字卡。下面附上完整剧本和分镜表。

请输出（中文，900 字以内）：
1. **最重要的 5 个弱点**，按影响排序，每个都附一个具体、可执行的改法。
2. **一个大胆的点子**：在上述限制内，让这部片子变得让人忘不掉。
3. **字卡审读**：哪句中文或英文写得弱、生硬或说得太满，请直接给出改写（中英都给）。
4. **应该剪掉什么**。

不要客套，泛泛的夸奖没有用。
"""


def bundle(header):
    parts = [header, "\n---\n\n# 附件一：剧本\n\n", (FILM / "SCREENPLAY.md").read_text(),
             "\n---\n\n# 附件二：分镜表\n\n", (FILM / "SHOTLIST.md").read_text()]
    return "".join(parts)


if __name__ == "__main__":
    (HERE / "gemini_bundle.md").write_text(bundle(GEMINI))
    (HERE / "grok_bundle.md").write_text(bundle(GROK))
    print("wrote gemini_bundle.md, grok_bundle.md")
