
PROMPT_E_1 = """{text}

Above are two argumentative texts discussing the same topic with the same stance.

Your task is to determine **which argumentative text evokes stronger emotions in you**. There are three label options:
- 0: Both arguments evoke a similar level of emotion, or neither argument evokes emotions in you.)
- 1: Argument 1 evokes stronger emotions in you.)
- 2: Argument 2 evokes stronger emotions in you.)

Which label will you choose?

Answer: """

PROMPT_C_1  = """{text}

Above are two argumentative texts discussing the same topic with the same stance.

Your task is to evaluate each pair to determine **which argumentative text you find more convincing**. There are three label options:
- 0 (Both arguments are equally convincing.)
- 1 (Argument 1 is more convincing.)
- 2 (Argument 2 is more convincing.)

Which label will you choose?

Answer: """



PROMPT_E_2 = """Below, you will find one pair of argumentative texts discussing the same topic with the same stance. The topic may be a binary choice, a bill from UK parliamentary debates, or a simple statement. Both arguments either support or oppose the topic, or they favor one side if the topic involves a binary choice.

Your task is to evaluate each pair to determine **which argumentative text evokes stronger emotions in you**. There are three label options:
- 0 (Both arguments evoke a similar level of emotion, or neither argument evokes emotions in you.)
- 1 (Argument 1 evokes stronger emotions in you.)
- 2 (Argument 2 evokes stronger emotions in you.)

{text}

Please answer the label **without** any explanations.

Answer: """

PROMPT_C_2  = """Below, you will find one pair of argumentative texts discussing the same topic with the same stance. The topic may be a binary choice, a bill from UK parliamentary debates, or a simple statement. Both arguments either support or oppose the topic, or they favor one side if the topic involves a binary choice.

Your task is to evaluate each pair to determine **which argumentative text you find more convincing**. There are three label options:
- 0 (Both arguments are equally convincing.)
- 1 (Argument 1 is more convincing.)
- 2 (Argument 2 is more convincing.)

{text}

Please answer the label **without** any explanations.

Answer: """


PROMPT_E_3 = """You are presented with two arguments that both address the same issue from the same perspective.

{text}

Your task: Decide which of the two arguments evokes stronger emotions. Use the following labels:
0: Both arguments evoke a similar level of emotion, or neither argument evokes emotions in you.
1: Argument 1 evokes stronger emotions in you.
2: Argument 2 evokes stronger emotions in you.

Which label do you select?

I select label """

PROMPT_C_3  = """You are presented with two arguments that both address the same issue from the same perspective.

{text}

Your task: Decide which of the two arguments you find more compelling. Use the following labels:
0: Both arguments have equal convincing strength.
1: Argument 1 is more convincing.
2: Argument 2 is more convincing.

Which label do you select?

I select label """

prompte = """Below, you will find one pair of argumentative texts discussing the same topic with the same stance. The topic may be a binary choice, a bill from UK parliamentary debates, or a simple statement. Both arguments either support or oppose the topic, or they favor one side if the topic involves a binary choice.

Your task is to evaluate each pair to determine **which argumentative text evokes stronger emotions in you**. There are three label options:
0 (Both arguments evoke a similar level of emotion, or neither argument evokes emotions.)
1 (Argument 1 evokes stronger emotions.)
2 (Argument 2 evokes stronger emotions.)

**Note**: Truncated sentences or grammatical errors should be **ignored**.

Please answer your label option **without** any explanations.

{text}
"""
#
promptc = """Below, you will find one pair of argumentative texts discussing the same topic with the same stance. The topic may be a binary choice, a bill from UK parliamentary debates, or a simple statement. Both arguments either support or oppose the topic, or they favor one side if the topic involves a binary choice.

Your task is to evaluate each pair to determine **which argumentative text you find more convincing**. There are three label options:
0 (Both arguments are equally convincing.)
1 (Argument 1 is more convincing.)
2 (Argument 2 is more convincing.)

**Note**: Truncated sentences or grammatical errors should be **ignored**.

Please answer your label option **without** any explanations.

{text}
"""

prompte_explain = """Below, you will find one pair of argumentative texts discussing the same topic with the same stance. The topic may be a binary choice, a bill from UK parliamentary debates, or a simple statement. Both arguments either support or oppose the topic, or they favor one side if the topic involves a binary choice.

Your task is to evaluate each pair to determine **which argumentative text evokes stronger emotions in you**. There are three label options:
0 (Both arguments evoke a similar level of emotion, or neither argument evokes emotions.)
1 (Argument 1 evokes stronger emotions.)
2 (Argument 2 evokes stronger emotions.)

**Note**: Truncated sentences or grammatical errors should be **ignored**.

Please answer your label option and briefly explain why you choose this label.

{text}

Below is an example answer for you; please follow this format in your response.
Label: 1
Explanation: Argument 1 evokes stronger emotions in me, because it reminds me of my family.
"""
#
promptc_explain = """Below, you will find one pair of argumentative texts discussing the same topic with the same stance. The topic may be a binary choice, a bill from UK parliamentary debates, or a simple statement. Both arguments either support or oppose the topic, or they favor one side if the topic involves a binary choice.

Your task is to evaluate each pair to determine **which argumentative text you find more convincing**. There are three label options:
0 (Both arguments are equally convincing.)
1 (Argument 1 is more convincing.)
2 (Argument 2 is more convincing.)

**Note**: Truncated sentences or grammatical errors should be **ignored**.

Please answer your label option and briefly explain why you choose this label.

{text}

Below is an example answer for you; please follow this format in your response.
Label: 2
Explanation: because Argument 2 provides more statistics supporting the claim, while Argument 1 contains logical fallacies.
"""

prompte_explain2 = """Below, you will find one pair of argumentative texts discussing the same topic with the same stance. The topic may be a binary choice, a bill from UK parliamentary debates, or a simple statement. Both arguments either support or oppose the topic, or they favor one side if the topic involves a binary choice.

Your task is to evaluate each pair to determine **which argumentative text evokes stronger emotions in you**. There are three label options:
0 (Both arguments evoke a similar level of emotion, or neither argument evokes emotions.)
1 (Argument 1 evokes stronger emotions.)
2 (Argument 2 evokes stronger emotions.)

**Note**: Truncated sentences or grammatical errors should be **ignored**.

Please answer your label option and briefly explain why you choose this label.

{text}

Below is an example answer for you; please follow this format in your response.
Label: 2
Explanation: Argument 2 evokes stronger emotions in me, because I totally disagree with its point and feel disgust about what it says.
"""
#
promptc_explain2 = """Below, you will find one pair of argumentative texts discussing the same topic with the same stance. The topic may be a binary choice, a bill from UK parliamentary debates, or a simple statement. Both arguments either support or oppose the topic, or they favor one side if the topic involves a binary choice.

Your task is to evaluate each pair to determine **which argumentative text you find more convincing**. There are three label options:
0 (Both arguments are equally convincing.)
1 (Argument 1 is more convincing.)
2 (Argument 2 is more convincing.)

**Note**: Truncated sentences or grammatical errors should be **ignored**.

Please answer your label option and briefly explain why you choose this label.

{text}

Below is an example answer for you; please follow this format in your response.
Label: 1
Explanation: Argument 1 is more convincing, because I totally agree with its point and it evokes my empathy.
"""

#global prompt

prompt = {
"chat_1": (prompte, promptc),
"chat_2": (prompte_explain, promptc_explain),
"chat_3": (prompte_explain2, promptc_explain2)
}





