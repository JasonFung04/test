# 顾问任务：Gemini（科学顾问）与 Grok（剧本顾问）

片头和片尾的署名只写**实际参与**的工具。Gemini 和 Grok 的意见交回并被采纳后，才会出现在署名里。

## 怎么跑（在你本地，大约 5 分钟）

```bash
git clone https://github.com/JasonFung04/test.git && cd test
git checkout claude/cannes-award-short-film-srgdht
python3 film/consult/make_bundles.py        # 用最新剧本重新生成两个任务包（可选）

# Gemini CLI：科学顾问
gemini -p "$(cat film/consult/gemini_bundle.md)" > film/consult/gemini_notes.md

# Grok CLI：剧本顾问（如果你的 grok CLI 不支持 -p，就进入交互模式把文件内容粘进去）
grok -p "$(cat film/consult/grok_bundle.md)" > film/consult/grok_notes.md

git add film/consult/*_notes.md && git commit -m "顾问意见：Gemini 与 Grok" && git push
```

更简单的做法：打开 `gemini_bundle.md` 和 `grok_bundle.md`，把全文分别粘进 Gemini 和 Grok 的网页版，再把它们的回复直接贴给我。

## 之后会发生什么

我会逐条评估它们的意见，采纳的写进 `ADOPTED.md`（注明来源），再据此修改剧本和画面。署名按实际贡献写：

- Gemini：科学顾问 *Science Consultant*
- Grok：剧本顾问 *Script Consultant*
- Fable（Anthropic）：剧本医生 *Script Doctor*，已完成，见 `fable_review_v1.md`
