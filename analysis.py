from openai import OpenAI
import os
import re

def generate_feedback(transcript_text, question, role, additional_context, feedback_type):
    # Sends the transcript to gpt40mini to analyse, returns the feedback
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    cleaned_text = re.sub(r"\s+", " ", transcript_text).strip()
    
    if feedback_type == 'Detailed':
        prompt = f"""
        You are an expert interview coach analysing a candidate's video interview response. Address the candidate as 'you'.

        Question:
        {question}

        Role:
        {role}

        Transcript:
        {cleaned_text}

        Additional context:
        {additional_context}

        Provide feedback in this format:
        1. **Summary of the answer**
        2. **Strengths** (3 bullet points)
        3. **Areas to improve** (3 bullet points)
        4. **Filler words**
        5. **Actionable tips**
        6. Scores out of 10 for Clarity, Confidence, and Structure.
        7. **Conclusion**
        """
    else:
        prompt = f"""
        You are an expert interview coach analysing a candidate's video interview response. Address the candidate as 'you'.
        Provide a short couple sentence summary of the interview mentioning strengths shown and one area for improvement

        Question:
        {question}

        Role:
        {role}

        Transcript:
        {cleaned_text}

        Additional context:
        {additional_context}
        """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content