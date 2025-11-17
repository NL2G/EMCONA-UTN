# EMCONA-UTN
This repository provides the code and data associated with our papers from the DFG-funded project, EMCONA (The Interplay of Emotions and Convincingness in Arguments), conducted at UTN (the University of Technology Nuremberg).

## Publications
[10/2025] Our new paper about emotions, convincingness, and logical fallacies is now on arXiv 🥳
- Paper: [Emotionally Charged, Logically Blurred: AI-driven Emotional Framing Impairs Human Fallacy Detection](https://arxiv.org/abs/2510.09695)
- Code + Data: to be updated
- Abstract:
  > Logical fallacies are common in public communication and can mislead audiences; fallacious arguments may still appear convincing despite lacking soundness, because convincingness is inherently subjective. We present the first computational study of how emotional framing interacts with fallacies and convincingness, using large language models (LLMs) to systematically change emotional appeals in fallacious arguments. We benchmark eight LLMs on injecting emotional appeal into fallacious arguments while preserving their logical structures, then use the best models to generate stimuli for a human study. Our results show that LLM-driven emotional framing reduces human fallacy detection in F1 by 14.5% on average. Humans perform better in fallacy detection when perceiving enjoyment than fear or sadness, and these three emotions also correlate with significantly higher convincingness compared to neutral or other emotion states. Our work has implications for AI-driven emotional manipulation in the context of fallacious argumentation.

[08/2025] One paper accepted at EMNLP 2025 Main 👏
- Paper: [Argument Summarization and its Evaluation in the Era of Large Language Models](https://aclanthology.org/2025.emnlp-main.1797/)
- Code + Data: [https://github.com/NL2G/argsum](https://github.com/NL2G/argsum)
- Abstract:
  > Large Language Models (LLMs) have revolutionized various Natural Language Generation (NLG) tasks, including Argument Summarization (ArgSum), a key subfield of Argument Mining. This paper investigates the integration of state-of-the-art LLMs into ArgSum systems and their evaluation. In particular, we propose a novel prompt-based evaluation scheme, and validate it through a novel human benchmark dataset. Our work makes three main contributions: (i) the integration of LLMs into existing ArgSum systems, (ii) the development of two new LLM-based ArgSum systems, benchmarked against prior methods, and (iii) the introduction of an advanced LLM-based evaluation scheme. We demonstrate that the use of LLMs substantially improves both the generation and evaluation of argument summaries, achieving state-of-the-art results and advancing the field of ArgSum. We also show that among the four LLMs integrated in (i) and (ii), Qwen-3-32B, despite having the fewest parameters, performs best, even surpassing GPT-4o.

[05/2025] One paper accepted at ACL 2025 Findings ✌️
- Paper: [Do Emotions Really Affect Argument Convincingness? A Dynamic Approach with LLM-based Manipulation Check](https://aclanthology.org/2025.findings-acl.1250/).
- Code + Data: [/llm_manipulation_check](/llm_manipulation_check)
- Abstract:
  > Emotions have been shown to play a role in argument convincingness, yet this aspect is underexplored in the natural language processing (NLP) community. Unlike prior studies that use static analyses, focus on a single text domain or language, or treat emotion as just one of many factors, we introduce a dynamic framework inspired by manipulation checks commonly used in psychology and social science; leveraging LLM-based manipulation checks, this framework examines the extent to which perceived emotional intensity influences perceived convincingness. Through human evaluation of arguments across different languages, text domains, and topics, we find that in over half of cases, human judgments of convincingness remain unchanged despite variations in perceived emotional intensity; when emotions do have an impact, they more often enhance rather than weaken convincingness.We further analyze whether 11 LLMs behave like humans in the same scenario, finding that while LLMs generally mirror human patterns,they struggle to capture nuanced emotional effects in individual judgments.

