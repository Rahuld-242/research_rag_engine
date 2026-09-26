import re
from pathlib import PurePosixPath


def get_topic_parts(document_id: str) -> list[str]:
    path = PurePosixPath(document_id)
    return list(path.parent.parts)

def _clean_topic_name(topic_part: str) -> str:
    without_prefix = re.sub(r"^\d+_", "", topic_part)
    cleaned_topic = without_prefix.replace("_", " ")
    return cleaned_topic

def clean_topic_names(topic_parts: list[str]) -> list[str]:
    cleaned_topics=[_clean_topic_name(topic_part) for topic_part in topic_parts]
    return cleaned_topics

def extract_topic_metadata(document_id: str) -> dict:
    topic_parts = get_topic_parts(document_id)
    topic_names = clean_topic_names(topic_parts)
    
    primary_topic = topic_names[0] if len(topic_names) >=1 else None
    subtopic = topic_names[1] if len(topic_names) >=2 else None
    
    topic_path = " > ".join(topic_names) if topic_names else None
    
    return {
        "topic_parts" : topic_parts,
        "topic_names" : topic_names,
        "primary_topic" : primary_topic,
        "subtopic" : subtopic,
        "topic_path" : topic_path
    }

#topic_parts=get_topic_parts("03_Fine_Tuning_and_Alignment/02_Instruction_Tuning_and_Preference_Optimization/DPO.pdf")
#print(clean_topic_names(topic_parts))

print(extract_topic_metadata("03_Fine_Tuning_and_Alignment/02_Instruction_Tuning_and_Preference_Optimization/DPO.pdf"))
print(extract_topic_metadata("RAG Survey.pdf"))