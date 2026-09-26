from research_rag.ingestion.metadata import (
    clean_topic_names,
    extract_topic_metadata,
    get_topic_parts,
)


def test_get_topic_parts_returns_parent_folder_parts():
    document_id = (
        "03_Fine_Tuning_and_Alignment/"
        "02_Instruction_Tuning_and_Preference_Optimization/"
        "DPO.pdf"
    )

    topic_parts = get_topic_parts(document_id)

    assert topic_parts == [
        "03_Fine_Tuning_and_Alignment",
        "02_Instruction_Tuning_and_Preference_Optimization",
    ]


def test_get_topic_parts_returns_empty_list_for_root_level_file():
    document_id = "RAG Survey.pdf"

    topic_parts = get_topic_parts(document_id)

    assert topic_parts == []


def test_clean_topic_names_removes_number_prefix_and_underscores():
    topic_parts = [
        "03_Fine_Tuning_and_Alignment",
        "02_Instruction_Tuning_and_Preference_Optimization",
    ]

    topic_names = clean_topic_names(topic_parts)

    assert topic_names == [
        "Fine Tuning and Alignment",
        "Instruction Tuning and Preference Optimization",
    ]


def test_extract_topic_metadata_returns_complete_topic_metadata():
    document_id = (
        "03_Fine_Tuning_and_Alignment/"
        "02_Instruction_Tuning_and_Preference_Optimization/"
        "DPO.pdf"
    )

    metadata = extract_topic_metadata(document_id)

    assert metadata == {
        "topic_parts": [
            "03_Fine_Tuning_and_Alignment",
            "02_Instruction_Tuning_and_Preference_Optimization",
        ],
        "topic_names": [
            "Fine Tuning and Alignment",
            "Instruction Tuning and Preference Optimization",
        ],
        "primary_topic": "Fine Tuning and Alignment",
        "subtopic": "Instruction Tuning and Preference Optimization",
        "topic_path": (
            "Fine Tuning and Alignment > "
            "Instruction Tuning and Preference Optimization"
        ),
    }


def test_extract_topic_metadata_handles_root_level_file():
    document_id = "RAG Survey.pdf"

    metadata = extract_topic_metadata(document_id)

    assert metadata == {
        "topic_parts": [],
        "topic_names": [],
        "primary_topic": None,
        "subtopic": None,
        "topic_path": None,
    }