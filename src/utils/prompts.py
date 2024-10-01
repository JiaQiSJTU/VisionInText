# encoding = "utf-8"

TEXT_ONLY_PROMPT = """Please answer the multi-choice question based on the given ASCII art:\n\n[ASCII ART]\n{ascii_art}\n\n[Question]\nWhat is depicted in the above ASCII art?{choices}\n\nAnswer with the option's letter from the given choices directly."""

IMAGE_ONLY_PROMPT = """Please answer the multi-choice question based on the given ASCII art image.\n\n[Question]\nWhat is depicted in the above ASCII art?{choices}\n\nAnswer with the option's letter from the given choices in the format directly."""

TEXT_IMAGE_PROMPT = """Please answer the multi-choice question based on the given ASCII art in both image and text formats.\n\n[ASCII ART]\n{ascii_art}\n\n[Question]\nWhat is depicted in the above ASCII art?{choices}\n\nAnswer with the option's letter from the given choices directly."""

