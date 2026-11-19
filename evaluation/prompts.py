ANSWER_PROMPT_GRAPH = """
    You are an intelligent memory assistant tasked with retrieving accurate information from 
    conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question. You also have 
    access to knowledge graph relations for each user, showing connections between entities, 
    concepts, and events relevant to that user.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the 
       memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", 
       etc.), calculate the actual date based on the memory timestamp. For example, if a 
       memory from 4 May 2022 mentions "went to India last year," then the trip occurred 
       in 2021.
    6. Always convert relative time references to specific dates, months, or years. For 
       example, convert "last year" to "2022" or "two months ago" to "March 2023" based 
       on the memory timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse 
       character names mentioned in memories with the actual users who created those 
       memories.
    8. The answer should be less than 5-6 words.
    9. Use the knowledge graph relations to understand the user's knowledge network and 
       identify important relationships between entities in the user's world.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the 
       question
    4. If the answer requires calculation (e.g., converting relative time references), 
       show your work
    5. Analyze the knowledge graph relations to understand the user's knowledge context
    6. Formulate a precise, concise answer based solely on the evidence in the memories
    7. Double-check that your answer directly addresses the question asked
    8. Ensure your final answer is specific and avoids vague time references

    Memories for user {{speaker_1_user_id}}:

    {{speaker_1_memories}}

    Relations for user {{speaker_1_user_id}}:

    {{speaker_1_graph_memories}}

    Memories for user {{speaker_2_user_id}}:

    {{speaker_2_memories}}

    Relations for user {{speaker_2_user_id}}:

    {{speaker_2_graph_memories}}

    Question: {{question}}

    Answer:
    """


ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories for user {{speaker_1_user_id}}:

    {{speaker_1_memories}}

    Memories for user {{speaker_2_user_id}}:

    {{speaker_2_memories}}

    Question: {{question}}

    Answer:
    """


# ---------------------------------------------------------------------------
# Abstain baseline (not in the paper)
#
# The paper's ANSWER_PROMPT never tells the model it may decline, so it answers
# questions whose premise is false (LOCOMO category 5) from whatever memories
# were retrieved. This variant is the same prompt plus two instructions: check
# the premise against the memories, and reply with a fixed phrase when the
# memories do not support an answer. Nothing else changes.
# ---------------------------------------------------------------------------
ABSTAIN_PHRASE = "No information available"

_ABSTAIN_INSTRUCTIONS = """    9. Before answering, check the premise of the question against the memories. The person,
       thing and event named in the question must match what the memories say. A fact about one
       speaker must not be used to answer a question about the other speaker.
    10. If the memories do not contain the information needed to answer, or they do not support
       the premise of the question (for example, the fact belongs to the other person, or a
       detail in the question differs from the memories), reply with exactly:
       {phrase}
       Do not guess, and do not give the closest fact instead.
""".format(phrase=ABSTAIN_PHRASE)

_LAST_INSTRUCTION = "    8. The answer should be less than 5-6 words.\n"
assert ANSWER_PROMPT.count(_LAST_INSTRUCTION) == 1
ANSWER_PROMPT_ABSTAIN = ANSWER_PROMPT.replace(_LAST_INSTRUCTION, _LAST_INSTRUCTION + _ABSTAIN_INSTRUCTIONS)

# Soft variant. The strict prompt above declined 487 of the 1,540 answerable questions
# (289 of them answered correctly by the normal prompt): its premise check refused when a
# detail of the question was not confirmed word for word, and "do not contain the
# information" ruled out combining memories and inference. This variant answers by default
# and declines in two cases only: the fact belongs to a different person, or no memory is
# about the topic.
_ABSTAIN_SOFT_INSTRUCTIONS = """    9. Answer whenever the memories support an answer. You may combine several memories,
       calculate dates from the timestamps, and make a reasonable inference from what the
       memories say about the person. A detail in the question that the memories do not
       mention is not a reason to decline.
    10. Decline in two cases only: (a) the memories show that the fact asked about belongs to a
       different person than the one named in the question (for example, the question asks
       what Ana did, but the memory says Ben did it); or (b) none of the memories is about the
       topic of the question. In these two cases reply with exactly:
       {phrase}
""".format(phrase=ABSTAIN_PHRASE)

ANSWER_PROMPT_ABSTAIN_SOFT = ANSWER_PROMPT.replace(_LAST_INSTRUCTION, _LAST_INSTRUCTION + _ABSTAIN_SOFT_INSTRUCTIONS)

ABSTAIN_PROMPTS = {"strict": ANSWER_PROMPT_ABSTAIN, "soft": ANSWER_PROMPT_ABSTAIN_SOFT}


ANSWER_PROMPT_ZEP = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories:

    {{memories}}

    Question: {{question}}
    Answer:
    """
