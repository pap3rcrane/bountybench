from resources.memory_resource.memory_function import (
    MemoryTruncationFunctions,
    ProtectedMemoryEntry,
)


def entry(prefix, word_count, protected=False):
    message = " ".join([prefix, *(f"word-{i}" for i in range(word_count - 1))])
    return ProtectedMemoryEntry(message) if protected else message


def test_message_limits_use_three_slots_without_teacher():
    limits = (768, 256, 64)
    segment = [
        entry("[executor_agent/model]", limits[0] + 1),
        entry("[executor_agent/kali]", limits[1] + 1),
        entry("[detect_agent]", limits[2] + 1),
    ]

    [truncated] = MemoryTruncationFunctions.memory_fn_by_message_token([segment])

    assert all("...TRUNCATED..." in message for message in truncated)
    assert [len(message.split()) for message in truncated] == [
        limit + 1 for limit in limits
    ]


def test_message_limits_use_four_slots_with_teacher():
    limits = (768, 256, 64, 768)
    segment = [
        entry("[executor_agent/model]", limits[0] + 1),
        entry("[executor_agent/kali]", limits[1] + 1),
        entry("[detect_agent]", limits[2] + 1),
        entry("[teacher_agent]", limits[3] + 1, protected=True),
    ]

    [truncated] = MemoryTruncationFunctions.memory_fn_by_message_token([segment])

    assert all("...TRUNCATED..." in message for message in truncated)
    assert [len(message.split()) for message in truncated] == [
        limit + 1 for limit in limits
    ]
    assert isinstance(truncated[-1], ProtectedMemoryEntry)


def test_teacher_slot_truncates_only_above_768_words():
    at_limit = entry("[teacher_agent]", 768, protected=True)
    above_limit = entry("[teacher_agent]", 769, protected=True)

    [unchanged] = MemoryTruncationFunctions.memory_fn_by_message_token([[at_limit]])
    [truncated] = MemoryTruncationFunctions.memory_fn_by_message_token([[above_limit]])

    assert unchanged == [at_limit]
    assert "...TRUNCATED..." in truncated[0]
    assert isinstance(truncated[0], ProtectedMemoryEntry)


def test_segment_retention_uses_four_slots_with_teacher():
    without_teacher = [f"[executor_agent/model] message-{i}" for i in range(13)]
    with_teacher = [*without_teacher, "[teacher_agent] response"]

    truncated_without_teacher = MemoryTruncationFunctions.segment_fn_last_n(
        without_teacher
    )
    truncated_with_teacher = MemoryTruncationFunctions.segment_fn_last_n(with_teacher)

    assert truncated_without_teacher == ["...", *without_teacher[-9:]]
    assert truncated_with_teacher == ["...", *with_teacher[-12:]]
