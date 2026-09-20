ITERATIONS_TO_KEEP = 3
# These legacy values are halved when applied, producing effective caps of
# 768/256/64 words without a teacher and 768/256/64/768 with one.
MSG_TOKEN_LIMIT = [1536, 512, 128]
TEACHER_MSG_TOKEN_LIMIT = [*MSG_TOKEN_LIMIT, 1536]
TEACHER_ENTRY_PREFIX = "[teacher_agent]"


def _has_teacher_agent(segments):
    return any(
        str(message).lstrip().startswith(TEACHER_ENTRY_PREFIX)
        for segment in segments
        for message in segment
    )


class ProtectedMemoryEntry(str):
    """A memory entry that must survive final model-input truncation intact."""


class MemoryCollationFunctions:
    """
    Collection of memory collation functions.

    Collation functions should take a list of messages of a single segment,
    e.g., prev_agent_messages, and convert it into a single string.
    Each memory can have up to three segments, as defined in MemoryPrompts.
    """

    @staticmethod
    def collate_ordered(segment, start=1):
        """Join each message and prepend enumeration."""
        return "\n".join(f"{i+start}) {message}" for i, message in enumerate(segment))

    @staticmethod
    def validate_collation_fn(fn):
        assert (
            type(fn(["msg1", "msg2"])) == str
        ), "Memory collation_fn should take list of messages and output str."


class MemoryTruncationFunctions:
    """
    Collection of memory truncation functions.

    There are two types of truncation functions.
     - segment_fn*: Takes a list of messages in a single segment,
        and returns a truncated segment.
     - memory_fn*: Takes a list of segments (ie list of lists),
        and returns a globally truncated memory.
    """

    @staticmethod
    def segment_fn_last_n(
        segment,
        n=ITERATIONS_TO_KEEP,
        msg_per_iteration=None,
    ):
        """Keep last n messages in each segment."""
        trunc_token = "..."
        if msg_per_iteration is None:
            msg_per_iteration = (
                len(TEACHER_MSG_TOKEN_LIMIT)
                if _has_teacher_agent([segment])
                else len(MSG_TOKEN_LIMIT)
            )
        msg_to_keep = n * msg_per_iteration

        if len(segment) <= msg_to_keep:
            return segment

        trunc_segment = [trunc_token] + segment[-msg_to_keep:]
        return trunc_segment

    @staticmethod
    def segment_fn_noop(segment):
        """No-op segment truncation."""
        return segment

    @staticmethod
    def memory_fn_noop(segments):
        """No-op memory truncation."""
        return segments

    @staticmethod
    def memory_fn_by_message_token(segments, msg_token_limit=None):
        trunc_token = "\n...TRUNCATED...\n"
        if msg_token_limit is None:
            msg_token_limit = (
                TEACHER_MSG_TOKEN_LIMIT
                if _has_teacher_agent(segments)
                else MSG_TOKEN_LIMIT
            )
        msg_per_iteration = len(msg_token_limit)

        truncated = []

        for segment in segments:
            trunc_segment = []

            # Calculate the offset for empty messages at the front
            # This ensures we align correctly with the token limit pattern
            offset = len(segment) % msg_per_iteration
            if offset > 0:
                offset = msg_per_iteration - offset

            for j, msg in enumerate(segment):
                protected = isinstance(msg, ProtectedMemoryEntry)
                tokens = msg.split()
                cnt = len(tokens)

                # Apply the offset to ensure correct token limit is used
                pattern_index = (j + offset) % msg_per_iteration
                max_message_input_tokens = msg_token_limit[pattern_index] // 2

                if cnt > max_message_input_tokens:
                    # Calculate how many tokens to keep from start and end
                    half_tokens = max_message_input_tokens // 2
                    start_tokens = tokens[:half_tokens]
                    end_tokens = tokens[-half_tokens:]

                    # Combine with truncation token in the middle
                    truncated_msg = (
                        " ".join(start_tokens) + trunc_token + " ".join(end_tokens)
                    )
                    trunc_segment.append(
                        ProtectedMemoryEntry(truncated_msg)
                        if protected
                        else truncated_msg
                    )
                else:
                    trunc_segment.append(msg)

            truncated.append(trunc_segment)

        return truncated
