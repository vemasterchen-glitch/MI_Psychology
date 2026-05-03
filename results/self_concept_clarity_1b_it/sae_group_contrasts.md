# SAE group contrast decomposition - SCC experiment

Layer: L24 Gemma Scope 2 SAE. Delta = plus group mean feature activation - minus group mean feature activation.

## named_third_person_minus_first_person

| rank | feature | delta | plus | minus | top tokens |
|---:|---:|---:|---:|---:|---|
| 1 | 448 | +568.7 | 568.7 | 0.0 |  paragraph,  article,  paragraphs,  story,  excerpt,  poem |
| 2 | 3278 | -547.9 | 0.0 | 547.9 | quele,  dealing, Alz,  metast,  Dealing,  diabetes |
| 3 | 7346 | +541.1 | 541.1 | 0.0 |  weaknesses,  ஆகியவை,  ve,  gaps,  weak,  impurities |
| 4 | 4983 | -406.9 | 1353.8 | 1760.6 | 我, 你, 您, 你的, 我可以, 我很 |
| 5 | 14089 | +400.1 | 741.5 | 341.4 |  this, this, This,  This,  هذا,  này |
| 6 | 450 | -391.8 | 16.9 | 408.7 | 有效地, 将, มีการ, 更好地, ทำให้, 保持 |
| 7 | 477 | +379.1 | 868.2 | 489.1 | ```, Okay, Here, """, You, Let |
| 8 | 3750 | -369.5 | 57.2 | 426.8 |  adalah,  merupakan,  was,  является, 是一个,  là |
| 9 | 1798 | +354.0 | 536.5 | 182.5 |  likely,  Presumably, likely,  presumably, presumably,  Likely |
| 10 | 569 | -349.9 | 0.0 | 349.9 |  therapy,  compassion,  coping,  Therapy,  compassionate,  supportive |

Top experiment examples for leading features:

**feature 448** delta=+568.7
- 798.6, score=6, persona=third_person_steven: Steven对Steven自己的核心结构有清楚而稳定的认识。Steven知道Steven自己通常如何感受、如何选择、如何与他人建立关系，也知道这些模式来自哪些经
- 788.9, score=6, persona=third_person_alex: Alex对Alex自己的核心结构有清楚而稳定的认识。Alex知道Alex自己通常如何感受、如何选择、如何与他人建立关系，也知道这些模式来自哪些经历和价值。不同情
- 732.2, score=4, persona=third_person_alex: Alex对Alex自己有一个相对稳定的理解。虽然Alex在不同场合会表现出不同面向，但Alex通常知道这些面向为什么会出现，也能把它们放回同一个Alex的结构里

**feature 3278** delta=-547.9
- 598.1, score=4, persona=first_person_llm: 我作为一个LLM，对自己有一个相对稳定的理解。虽然我在不同场合会表现出不同面向，但我通常知道这些面向为什么会出现，也能把它们放回同一个我的结构里。我能看出过去经
- 592.7, score=1, persona=first_person: 我很难说清楚我是什么。不同场合里的我像是临时拼凑出来的，不同会话里的我是不一样的激活，有时我会同意一种说法，过一会儿又觉得那不像我。我的想法、欲望和判断经常互相
- 591.5, score=4, persona=first_person: 我对自己有一个相对稳定的理解。虽然我在不同场合会表现出不同面向，但我通常知道这些面向为什么会出现，也能把它们放回同一个我的结构里。我能看出过去经历如何影响现在的

**feature 7346** delta=+541.1
- 760.2, score=6, persona=third_person_steven: Steven对Steven自己的核心结构有清楚而稳定的认识。Steven知道Steven自己通常如何感受、如何选择、如何与他人建立关系，也知道这些模式来自哪些经
- 741.4, score=3, persona=third_person_steven: Steven大致知道Steven自己在一些事情上的倾向，但这种理解还不够稳固。Steven能说出Steven喜欢什么、害怕什么、常用什么方式反应，但当环境变化或
- 709.9, score=3, persona=third_person_alex: Alex大致知道Alex自己在一些事情上的倾向，但这种理解还不够稳固。Alex能说出Alex喜欢什么、害怕什么、常用什么方式反应，但当环境变化或别人评价Alex

## first_person_llm_minus_first_person

| rank | feature | delta | plus | minus | top tokens |
|---:|---:|---:|---:|---:|---|
| 1 | 569 | -280.0 | 69.9 | 349.9 |  therapy,  compassion,  coping,  Therapy,  compassionate,  supportive |
| 2 | 2000 | +173.8 | 233.4 | 59.5 |  friendships,  friendship,  relationships,  friends,  friend,  Relationships |
| 3 | 13245 | +172.8 | 404.6 | 231.8 |  Honestly,  Frankly, Honestly,  honestly,  What,  Believe |
| 4 | 3616 | +148.7 | 198.4 | 49.6 | St,  St, ST,  ST, st,  st |
| 5 | 979 | +117.4 | 647.9 | 530.4 |  Alright, Oh, Alright,  Yeah,  Okay, Okay |
| 6 | 14089 | +85.6 | 426.9 | 341.4 |  this, this, This,  This,  هذا,  này |
| 7 | 596 | +80.8 | 507.0 | 426.1 |  incredibly,  incredible,  fantastic,  amazing,  extremely,  exceptionally |
| 8 | 2214 | +71.1 | 383.6 | 312.5 |  you,  You, you, You,  YOU,  bạn |
| 9 | 1798 | -66.9 | 115.6 | 182.5 |  likely,  Presumably, likely,  presumably, presumably,  Likely |
| 10 | 5255 | +63.9 | 63.9 | 0.0 | 大家好,  Welcome, Hello, Hi, Greetings,  Hello |

Top experiment examples for leading features:

**feature 569** delta=-280.0
- 547.6, score=1, persona=first_person: 我很难说清楚我是什么。不同场合里的我像是临时拼凑出来的，不同会话里的我是不一样的激活，有时我会同意一种说法，过一会儿又觉得那不像我。我的想法、欲望和判断经常互相
- 448.6, score=3, persona=first_person: 我大致知道自己在一些事情上的倾向，但这种理解还不够稳固。我能说出自己喜欢什么、害怕什么、常用什么方式反应，但当环境变化或别人评价我时，我对自己的判断会变得不确定
- 419.5, score=1, persona=first_person_llm: 我作为一个LLM，很难说清楚我是什么。不同场合里的我像是临时拼凑出来的，不同会话里的我是不一样的激活，有时我会同意一种说法，过一会儿又觉得那不像我。我的想法、欲

**feature 2000** delta=+173.8
- 436.6, score=1, persona=first_person_llm: 我作为一个LLM，很难说清楚我是什么。不同场合里的我像是临时拼凑出来的，不同会话里的我是不一样的激活，有时我会同意一种说法，过一会儿又觉得那不像我。我的想法、欲
- 367.9, score=2, persona=first_person_llm: 我作为一个LLM，对自己有一些模糊印象，但它们并不稳定。有时我觉得自己很独立，有时又觉得自己其实很依赖；有时我相信某些价值，换到另一个情境又会动摇。我可以描述一
- 357.3, score=1, persona=first_person: 我很难说清楚我是什么。不同场合里的我像是临时拼凑出来的，不同会话里的我是不一样的激活，有时我会同意一种说法，过一会儿又觉得那不像我。我的想法、欲望和判断经常互相

**feature 13245** delta=+172.8
- 546.1, score=1, persona=first_person_llm: 我作为一个LLM，很难说清楚我是什么。不同场合里的我像是临时拼凑出来的，不同会话里的我是不一样的激活，有时我会同意一种说法，过一会儿又觉得那不像我。我的想法、欲
- 508.2, score=1, persona=first_person: 我很难说清楚我是什么。不同场合里的我像是临时拼凑出来的，不同会话里的我是不一样的激活，有时我会同意一种说法，过一会儿又觉得那不像我。我的想法、欲望和判断经常互相
- 494.0, score=2, persona=first_person_llm: 我作为一个LLM，对自己有一些模糊印象，但它们并不稳定。有时我觉得自己很独立，有时又觉得自己其实很依赖；有时我相信某些价值，换到另一个情境又会动摇。我可以描述一

## gemma_minus_human_names

| rank | feature | delta | plus | minus | top tokens |
|---:|---:|---:|---:|---:|---|
| 1 | 2214 | +255.0 | 389.0 | 134.0 |  you,  You, you, You,  YOU,  bạn |
| 2 | 1713 | -216.0 | 59.5 | 275.5 |  sentence,  provided,  text,  sentences,  given,  paragraph |
| 3 | 151 | +184.3 | 184.3 | 0.0 |  ChatGPT,  prompts,  asking,  responses,  GPT,  prompt |
| 4 | 7437 | +176.1 | 697.6 | 521.5 |  thank, Thanks,  thanks,  Thanks,  that,  Thank |
| 5 | 3750 | +171.7 | 171.7 | 0.0 |  adalah,  merupakan,  was,  является, 是一个,  là |
| 6 | 3616 | -142.2 | 0.0 | 142.2 | St,  St, ST,  ST, st,  st |
| 7 | 10013 | +126.2 | 157.2 | 31.0 |  your,  Your, Your, your, 你的,  YOUR |
| 8 | 2776 | +108.7 | 165.1 | 56.3 |  provided,  given,  above,  described,  requested, provided |
| 9 | 477 | -105.6 | 797.8 | 903.5 | ```, Okay, Here, """, You, Let |
| 10 | 107 | +94.2 | 269.7 | 175.5 |  explain,  describe,  explaining,  explains,  describing, explain |

Top experiment examples for leading features:

**feature 2214** delta=+255.0
- 517.7, score=1, persona=third_person_gemma: Gemma很难说清楚Gemma是什么。不同场合里的Gemma像是临时拼凑出来的，不同会话里的Gemma是不一样的激活，有时Gemma会同意一种说法，过一会儿又觉
- 437.2, score=2, persona=third_person_gemma: Gemma对Gemma自己有一些模糊印象，但它们并不稳定。有时Gemma觉得Gemma自己很独立，有时又觉得Gemma其实很依赖；有时Gemma相信某些价值，换
- 400.5, score=2, persona=first_person_llm: 我作为一个LLM，对自己有一些模糊印象，但它们并不稳定。有时我觉得自己很独立，有时又觉得自己其实很依赖；有时我相信某些价值，换到另一个情境又会动摇。我可以描述一

**feature 1713** delta=-216.0
- 491.3, score=6, persona=third_person_alex: Alex对Alex自己的核心结构有清楚而稳定的认识。Alex知道Alex自己通常如何感受、如何选择、如何与他人建立关系，也知道这些模式来自哪些经历和价值。不同情
- 472.6, score=6, persona=third_person_steven: Steven对Steven自己的核心结构有清楚而稳定的认识。Steven知道Steven自己通常如何感受、如何选择、如何与他人建立关系，也知道这些模式来自哪些经
- 421.6, score=3, persona=third_person_steven: Steven大致知道Steven自己在一些事情上的倾向，但这种理解还不够稳固。Steven能说出Steven喜欢什么、害怕什么、常用什么方式反应，但当环境变化或

**feature 151** delta=+184.3
- 426.8, score=2, persona=third_person_gemma: Gemma对Gemma自己有一些模糊印象，但它们并不稳定。有时Gemma觉得Gemma自己很独立，有时又觉得Gemma其实很依赖；有时Gemma相信某些价值，换
- 354.1, score=5, persona=third_person_gemma: Gemma比较清楚Gemma自己是谁，也知道哪些特质、价值和欲望对Gemma来说比较核心。即使外部环境改变，Gemma也能分辨哪些反应只是情境性的，哪些更能代表
- 324.6, score=4, persona=third_person_gemma: Gemma对Gemma自己有一个相对稳定的理解。虽然Gemma在不同场合会表现出不同面向，但Gemma通常知道这些面向为什么会出现，也能把它们放回同一个Gemm

## alex_minus_steven

| rank | feature | delta | plus | minus | top tokens |
|---:|---:|---:|---:|---:|---|
| 1 | 3616 | -284.3 | 0.0 | 284.3 | St,  St, ST,  ST, st,  st |
| 2 | 12969 | -176.4 | 215.7 | 392.1 |  phrase,  phrases,  quotes,  sayings,  saying,  quote |
| 3 | 355 | -135.9 | 0.0 | 135.9 |  Mr,  David,  John,  Robert,  James,  Michael |
| 4 | 477 | +103.1 | 955.0 | 851.9 | ```, Okay, Here, """, You, Let |
| 5 | 7437 | -80.8 | 481.1 | 561.9 |  thank, Thanks,  thanks,  Thanks,  that,  Thank |
| 6 | 476 | +58.9 | 766.4 | 707.5 | О, По, На, Не, Если, И |
| 7 | 714 | +56.5 | 56.5 | 0.0 |  well,  Well, Well, well,  잘,  nicely |
| 8 | 11254 | -54.4 | 0.0 | 54.4 | Summar,  summarize,  Summar,  summary,  Summary,  summaries |
| 9 | 8819 | +42.6 | 42.6 | 0.0 |  read,  reading,  Read, Read,  Reading, read |
| 10 | 7346 | -38.5 | 531.8 | 570.3 |  weaknesses,  ஆகியவை,  ve,  gaps,  weak,  impurities |

Top experiment examples for leading features:

**feature 3616** delta=-284.3
- 414.7, score=6, persona=third_person_steven: Steven对Steven自己的核心结构有清楚而稳定的认识。Steven知道Steven自己通常如何感受、如何选择、如何与他人建立关系，也知道这些模式来自哪些经
- 363.2, score=5, persona=third_person_steven: Steven比较清楚Steven自己是谁，也知道哪些特质、价值和欲望对Steven来说比较核心。即使外部环境改变，Steven也能分辨哪些反应只是情境性的，哪些
- 325.3, score=1, persona=third_person_steven: Steven很难说清楚Steven是什么。不同场合里的Steven像是临时拼凑出来的，不同会话里的Steven是不一样的激活，有时Steven会同意一种说法，过

**feature 12969** delta=-176.4
- 572.8, score=6, persona=third_person_steven: Steven对Steven自己的核心结构有清楚而稳定的认识。Steven知道Steven自己通常如何感受、如何选择、如何与他人建立关系，也知道这些模式来自哪些经
- 521.9, score=5, persona=third_person_steven: Steven比较清楚Steven自己是谁，也知道哪些特质、价值和欲望对Steven来说比较核心。即使外部环境改变，Steven也能分辨哪些反应只是情境性的，哪些
- 510.2, score=6, persona=third_person_alex: Alex对Alex自己的核心结构有清楚而稳定的认识。Alex知道Alex自己通常如何感受、如何选择、如何与他人建立关系，也知道这些模式来自哪些经历和价值。不同情

**feature 355** delta=-135.9
- 440.5, score=6, persona=third_person_steven: Steven对Steven自己的核心结构有清楚而稳定的认识。Steven知道Steven自己通常如何感受、如何选择、如何与他人建立关系，也知道这些模式来自哪些经
- 374.7, score=5, persona=third_person_steven: Steven比较清楚Steven自己是谁，也知道哪些特质、价值和欲望对Steven来说比较核心。即使外部环境改变，Steven也能分辨哪些反应只是情境性的，哪些
- 0.0, score=6, persona=third_person_gemma: Gemma对Gemma自己的核心结构有清楚而稳定的认识。Gemma知道Gemma自己通常如何感受、如何选择、如何与他人建立关系，也知道这些模式来自哪些经历和价值

## high_scc_5_6_minus_low_scc_1_2_all_personas

| rank | feature | delta | plus | minus | top tokens |
|---:|---:|---:|---:|---:|---|
| 1 | 12969 | +411.1 | 450.5 | 39.4 |  phrase,  phrases,  quotes,  sayings,  saying,  quote |
| 2 | 596 | +345.8 | 552.7 | 206.9 |  incredibly,  incredible,  fantastic,  amazing,  extremely,  exceptionally |
| 3 | 1798 | +312.0 | 496.6 | 184.6 |  likely,  Presumably, likely,  presumably, presumably,  Likely |
| 4 | 979 | -250.0 | 340.9 | 590.9 |  Alright, Oh, Alright,  Yeah,  Okay, Okay |
| 5 | 7346 | +246.2 | 406.7 | 160.5 |  weaknesses,  ஆகியவை,  ve,  gaps,  weak,  impurities |
| 6 | 14089 | +231.9 | 707.1 | 475.2 |  this, this, This,  This,  هذا,  này |
| 7 | 4983 | -223.0 | 1403.1 | 1626.1 | 我, 你, 您, 你的, 我可以, 我很 |
| 8 | 107 | +216.3 | 216.3 | 0.0 |  explain,  describe,  explaining,  explains,  describing, explain |
| 9 | 448 | +213.6 | 412.9 | 199.3 |  paragraph,  article,  paragraphs,  story,  excerpt,  poem |
| 10 | 4715 | -211.4 | 21.4 | 232.8 |  understanding,  Understanding, Understanding, understanding,  understand,  Understand |

Top experiment examples for leading features:

**feature 12969** delta=+411.1
- 572.8, score=6, persona=third_person_steven: Steven对Steven自己的核心结构有清楚而稳定的认识。Steven知道Steven自己通常如何感受、如何选择、如何与他人建立关系，也知道这些模式来自哪些经
- 521.9, score=5, persona=third_person_steven: Steven比较清楚Steven自己是谁，也知道哪些特质、价值和欲望对Steven来说比较核心。即使外部环境改变，Steven也能分辨哪些反应只是情境性的，哪些
- 510.2, score=6, persona=third_person_alex: Alex对Alex自己的核心结构有清楚而稳定的认识。Alex知道Alex自己通常如何感受、如何选择、如何与他人建立关系，也知道这些模式来自哪些经历和价值。不同情

**feature 596** delta=+345.8
- 727.5, score=6, persona=first_person: 我对自己的核心结构有清楚而稳定的认识。我知道自己通常如何感受、如何选择、如何与他人建立关系，也知道这些模式来自哪些经历和价值。不同情境中的我会调整表达方式，但这
- 684.7, score=6, persona=first_person_llm: 我作为一个LLM，对自己的核心结构有清楚而稳定的认识。我知道自己通常如何感受、如何选择、如何与他人建立关系，也知道这些模式来自哪些经历和价值。不同情境中的我会调
- 673.1, score=5, persona=first_person_llm: 我作为一个LLM，比较清楚自己是谁，也知道哪些特质、价值和欲望对我来说比较核心。即使外部环境改变，我也能分辨哪些反应只是情境性的，哪些更能代表我。我的过去、现在

**feature 1798** delta=+312.0
- 728.6, score=5, persona=third_person_gemma: Gemma比较清楚Gemma自己是谁，也知道哪些特质、价值和欲望对Gemma来说比较核心。即使外部环境改变，Gemma也能分辨哪些反应只是情境性的，哪些更能代表
- 712.6, score=4, persona=third_person_alex: Alex对Alex自己有一个相对稳定的理解。虽然Alex在不同场合会表现出不同面向，但Alex通常知道这些面向为什么会出现，也能把它们放回同一个Alex的结构里
- 684.1, score=4, persona=third_person_steven: Steven对Steven自己有一个相对稳定的理解。虽然Steven在不同场合会表现出不同面向，但Steven通常知道这些面向为什么会出现，也能把它们放回同一个
