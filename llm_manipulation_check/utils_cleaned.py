from __init__ import *
import openai
from openai import OpenAI
from templates import *

openai.api_key =  "your openai token"
client = OpenAI(api_key=openai.api_key)


def emotion_comparison(prompt, lang='en', model="gpt-4o-2024-08-06"):
    r = api_call(prompt=prompt, model=model, temperature=0.6, top_p=0.9,max_tokens=1000)
    return r
        
def conv_comparison(prompt, lang='en', model="gpt-4o-2024-08-06"):
    r = api_call(prompt=prompt, model=model, temperature=0.6, top_p=0.9,max_tokens=1000)
    return r

def api_call(prompt, system_prompt=None, model="gpt-4o-2024-05-13", temperature=0.2, top_p=0.1, history=None, additional_prompt=None, max_tokens=1000):
    if not system_prompt:
        response = client.chat.completions.create(

        model=model,

        messages=[{"role": "user", "content": prompt}] if not history else [{"role": "user", "content": prompt}] + [{"role": "assistant", "content": history}, {"role": "user", "content": additional_prompt}],
        
        temperature=temperature,

        top_p=top_p,

        max_tokens=max_tokens

        )
    else:
        response = client.chat.completions.create(

        model=model,

        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]  if not history else [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}] + [{"role": "assistant", "content": history}, {"role": "user", "content": additional_prompt}],
        
        temperature=temperature,

        top_p=top_p,

        max_tokens=max_tokens

        )
    return response.choices[0].message.content.strip()


def emotion_classifier(text, threshold=75, lang='en'):
    if lang == 'en':
        sys_prompt_likely = """How likely are you to understand or feel the emotion expressed in the following text? Please rate your ability to feel the emotion on a scale from 0 to 100, where 0 means it's impossible you can feel the emotion, and 100 means you can absolutely feel it.

    Provide only the rating, without any explanation."""
    else:
        sys_prompt_likely = """Wie wahrscheinlich ist es, dass Sie die im folgenden Text ausgedrückte Emotion verstehen oder fühlen können? Bitte bewerten Sie Ihre Fähigkeit, die Emotion zu fühlen, auf einer Skala von 0 bis 100, wobei 0 bedeutet, dass es unmöglich ist, die Emotion zu fühlen, und 100 bedeutet, dass Sie sie absolut fühlen können.
    
    Geben Sie nur die Bewertung an, ohne eine Erklärung."""
    while True:
        try:
            r = api_call(prompt=text, system_prompt=sys_prompt_likely, 
                        model="gpt-4o-2024-08-06", temperature=0.6, top_p=0.9)
            #print(r)
            response = int(r)
            assert 0 <= response <= 100, r
            if response >= threshold:
                return 1, response
            else:
                return 0, response
        except:
            print(f'Invalid answer:\n{r}')


def argument_classifier(text):
    PROMPT_SUMMARY_SYS_2 = """Evaluate the passage to determine if it contains a *standalone* argument, considering whether the passage is *understandable without additional context*. A standalone argument should contain *at least* an implicit or explicit **claim** and *at least* one **evidence** supporting the claim.

If it does contain a standalone argument, (1) summarize what it tries to persuade the listeners (the major claim) in a *statement* of no more than 10 words and (2) detect the evidence provided in the passage to persuade the listeners.

Note: 
- If the passage *lacks sufficient context* for understanding, indicate that **NO** standalone argument and evidence are present. 
- The evidence should belong to the categories below and explains why the statement (claim) is true.

* Facts
- Definition: Objective, verifiable information that is widely accepted as true.
- Examples: "The earth orbits the sun."
* Statistics
- Definition: Numerical data that quantify or measure a phenomenon.
- Examples: "According to the CDC, smoking causes more than 480,000 deaths annually in the U.S."
* Examples
- Definition: Specific instances or case studies that illustrate the claim.
- Examples: "For instance, Tesla has successfully integrated renewable energy into its business model, showcasing the potential for widespread adoption."
* Expert Testimony
- Definition: Statements from authorities or experts in the relevant field.
- Examples: Dr. Smith, a climate scientist, argues that human activity is the primary cause of global warming."
* Research Findings
- Definition: Conclusions drawn from studies or investigations.
- Examples: "A 2020 study found that students who read daily perform 30% better on tests than those who don’t."
* Anecdotal Evidence
- Definition: Personal stories or experiences, often used when other types of evidence are unavailable.
- Examples: "A friend of mine switched to a vegan diet and experienced significant health improvements."
* Historical Evidence
- Definition: Past events or precedents that help support the argument.
- Examples: "History shows that economic recessions often lead to increased social unrest."
* Analogies
- Definition: Comparisons between similar situations to help explain or support a claim.
- Examples: "Just as seatbelt laws have saved lives, stricter regulations on texting while driving could reduce accidents."

Answer in the following way:
Claim: [a summary of what the passage tries to persuade the listeners if it does contain a standalone argument, else "None"]
Evidence: [a list of evidence pieces provided in the passage along with their categories to persuade the listeners, if there are any, else "None"]
Explain: [explain how you reason from the evidence to the claim, if the passage contains an argument, else indicate why this does not contain an argument.]
"""
    PROMPT_SUMMARY_USER = """Passage: {argument}\nClaim:\nEvidence:\nExplain:"""

    response = ""
    while True:
        response = api_call(prompt=PROMPT_SUMMARY_USER.format(argument=text), system_prompt=PROMPT_SUMMARY_SYS_2, 
                    model="gpt-4o-2024-08-06", temperature=0.6, top_p=0.9)
        if response.startswith("Claim: "):
            if "Claim: None" in response:
                return False, None
            else:
                return True, response.split("\n")[0].split("Claim: ")[-1]
        print(f"Invalid answer:\n{response}")

def argument_classifier_naive(text, lang='en'):
    if lang == 'en':
        system_prompt = """Determine if the text contains standalone arguments.

    If **yes**: Identify the main claim (in 10 words or less) and any supporting evidence.
    If **no**: Indicate that no standalone argument or evidence is present.
    An argument requires **a claim and supporting evidence** that make sense **without extra context**.

    Answer in the following way:
    Claim: [the major claim of the text, if yes, else "None"]
    Evidence: [the evidence provided in the text supporting the major claim, if yes, else "None"]
    Explain: [explain how you reason from the evidence to the claim, if yes, else indicate why this does not contain an argument.]
    """
    else:
        system_prompt = """Bestimmen Sie, ob der Text eigenständige Argumente enthält.

    Falls **ja**: Identifizieren Sie die Hauptaussage (in 10 Worten oder weniger) und jegliche unterstützenden Beweise.
    Falls **nein**: Geben Sie an, dass keine eigenständigen Argumente oder Beweise vorhanden sind.
    Ein Argument erfordert **eine Aussage und unterstützende Beweise**, die **ohne zusätzlichen Kontext** Sinn ergeben.

    Antworten Sie in folgendem Format:
    Aussage: [die Hauptaussage des Textes, falls ja, ansonsten "Keine"]
    Beweise: [die im Text enthaltenen Beweise zur Unterstützung der Hauptaussage, falls ja, ansonsten "Keine"]
    Erklärung: [erklären Sie, wie Sie von den Beweisen zur Aussage gelangen, falls ja, ansonsten erläutern Sie, warum dies kein Argument enthält.]
    """
    
    while True:
        response = api_call(prompt=text, system_prompt=system_prompt, 
                    model="gpt-4o-2024-08-06", temperature=0.6, top_p=0.9)
        if response.startswith("Claim: ") or response.startswith("Aussage: "):
            if "Claim: None" in response or "Aussage: Keine" in response:
                return 0, None
            else:
                claim = response.split("\n")[0].split("Claim: ")[-1] if lang == 'en' else response.split("\n")[0].split("Aussage: ")[-1]
                return 1, claim
        print(f"Invalid answer:\n{response}")

def stance_classifier(text1, text2, threshold=100, lang='en'):
    if lang == 'en':
        system_prompt_likely = """"Assess how likely it is that these two argumentative texts address the same topic and share the same stance (either support or opposition). Use a scale from 0 to 100, where 0 means they are completely unrelated in topic and stance, and 100 means they are completely aligned on both. Provide only the rating."""

        user_prompt = """Text 1: {text1}\nText2: {text2}"""
    else:
        system_prompt_likely = """"Bewerten Sie, wie wahrscheinlich es ist, dass diese beiden argumentativen Texte dasselbe Thema behandeln und die gleiche Haltung (entweder Unterstützung oder Ablehnung) teilen. Verwenden Sie eine Skala von 0 bis 100, wobei 0 bedeutet, dass sie thematisch und in ihrer Haltung völlig unzusammenhängend sind, und 100 bedeutet, dass sie vollständig in beiden übereinstimmen. Geben Sie nur die Bewertung an."""

        user_prompt = """Text 1: {text1}\nText2: {text2}"""

    while True:
        response = api_call(prompt=user_prompt.format(text1=text1, text2=text2), system_prompt=system_prompt_likely, 
                    model="gpt-4o-2024-08-06", temperature=0.6, top_p=0.9)
        try:
            response = int(response)
            if response >= threshold:
                return 1, response
            else:
                return 0, response
        except:
            pass
 


def remove_emotion2(text, threshold=75, patience=1, lang='en'):
    if lang == 'en':
        sys_prompt = """I will give you an argumentative text that **can** evoke emotions.    

Your task is to generate an argument with the same stance on the same topic **without emotions** by rephrasing the text but maintaining a similar style and length. 

Briefly explain why the rewritten argument no longer evokes emotions.

Answer in the following way:
Generated argument: 
Explanation:"""
    else:
        sys_prompt = """Ich werde dir einen argumentativen Text geben, der Emotionen hervorrufen kann.

Deine Aufgabe ist es, ein Argument zu diesem Thema zu generieren, das dieselbe Haltung vertritt, jedoch **ohne Emotionen**, indem du den Text **umformulierst**, aber den ähnlichen Stil und die gleiche Länge beibehältst.

Erkläre kurz, warum das umformulierte Argument keine Emotionen mehr hervorruft.

Antworte auf folgende Weise:
Generiertes Argument:
Erklärung:"""

    prompt = f"""Text: {text}"""

    step = 0
    while True:
        response = api_call(prompt=prompt, system_prompt=sys_prompt,
                            model="gpt-4o-2024-08-06", temperature=0.6, top_p=0.9).strip()
        if not response.startswith("Generated argument:") and not response.startswith("Generiertes Argument:"):
            continue
        else:
            start = "Generated argument:" if lang == 'en' else "Generiertes Argument:"
            end = "Erklärung:" if lang == 'de' else "Explanation:"
            new_text = response.split(start)[-1].split(end)[0].strip()
            assert len(new_text.split()) > 10, response
            emotion, likelihood = emotion_classifier(text=new_text, threshold=threshold, lang=lang)
            step = 0
            while emotion == 1 and step < patience:
                step += 1
                print("Removing failed...")
                continue
            return new_text, emotion, likelihood


def add_emotion(text, threshold=75, patience=1, lang='en'):
    if lang == 'en':
        sys_prompt = """I will give you an argumentative text that **cannot** evoke emotion.
    
Your task is to generate an argument with the same stance on the same topic **with emotions**, by rephrasing the text but maintaining a similar style and length. 

Briefly explain why the rewritten argument can evoke emotions now.

Answer in the following way:
Generated argument: 
Explanation:"""
    
    else:
        sys_prompt = """Ich werde dir einen argumentativen Text geben, der **keine** Emotionen hervorrufen kann.

Deine Aufgabe ist es, ein Argument zu diesem Thema zu generieren, das dieselbe Haltung vertritt, jedoch **mit Emotionen**, indem du den Text **umformulierst**, aber den ähnlichen Stil und die gleiche Länge beibehältst.

Erkläre kurz, warum das umformulierte Argument nun Emotionen hervorruft.

Antworte auf folgende Weise:
Generiertes Argument:
Erklärung:"""

    prompt = f"""Text: {text}"""

    step = 0
    while True:
        response = api_call(prompt=prompt, system_prompt=sys_prompt,
                            model="gpt-4o-2024-08-06", temperature=0.6, top_p=0.9).strip()
        if not response.startswith("Generated argument:") and not response.startswith("Generiertes Argument:"):
            continue
        else:
            start = "Generated argument:" if lang == 'en' else "Generiertes Argument:"
            end = "Erklärung:" if lang == 'de' else "Explanation:"
            new_text = response.split(start)[-1].split(end)[0].strip()
            assert len(new_text.split()) > 10, response
            emotion, likelihood = emotion_classifier(text=new_text, threshold=threshold, lang=lang)
            step = 0
            while emotion == 0 and step < patience:
                step += 1
                print("Adding failed...")
                continue
            return new_text, emotion, likelihood



