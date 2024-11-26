# script for processing data

import os
import csv
import json


def txt_to_tsv(file_path, output_path):
    """Script to process txt -> tsv files.

    Produces...

    Args:
        file_path str: File input path
        output_path str: Output directory for tsv files
    """

    final_output = []

    with open(file_path, "r") as txt_file:
        read_file = txt_file.read()
        # It splits the file into sections by section delim
        # NOTE: Lets improve this later
        sections = read_file.split(
            "***************************************************"
        )
        # Iterating through sections and combining the lines when they aren't blank
        for section in sections:
            lines = section.split("\n")
            lines = [
                l for l in lines if l.strip() != ""
            ]  # Represents actual line with text we want to refer to

            # Ensuring we have at least read the first line
            if len(lines) > 1:

                # concatentating the extra spacing
                id = lines[0].replace(" ", "")
                author_info = lines[1] + lines[2]
                author_info = author_info.replace(" ", "")

                # Settting a variable for when story starts & sotry end
                story_start = 3
                story_end = 0

                # enumerating through the lines in this section and assigning story end point when we reach ? #1
                for i, line in enumerate(lines):
                    if line.startswith("1"):
                        story_end = i  # index/line num when story actually ends!

                # Getting story text
                # There is apparently some checking on the story end point, I believe this is to handle blank stories
                if story_end == 3:
                    story_text = lines[3]
                    story_text.strip("\n")

                # Now that we have story start and finish, we can assign a variable to contain story text
                else:
                    story_text = lines[story_start:story_end]
                    # Stripping \n and joining the remaining text:
                    story_text = [line.strip("\n") for line in story_text]
                    story_text = (" ").join(story_text)

                # assigning remaining rows to the rest of the story
                remaining_rows = lines[story_end:]

                output = [id, author_info, story_text] + remaining_rows

                final_output.append(output)

    with open(f"../models/Data/{output_path}", "w") as tsv_file:
        file_reader = csv.writer(tsv_file, delimiter="\t")
        for row in final_output:
            file_reader.writerow(row)


def write_to_json(path, split):
    """Script to process tsv -> json files.

    Args:
        path str: Takes input path to our data directory as one arg
        split str: Split path for questions and answers section of resulting json

    Output:
        examples = list[dict]

    """
    # Paths for writing to json
    full_path = os.path.join(path, split)
    questions_path = f"{full_path}.tsv"
    # answers_path = f"MCTest/mc160.dev.ans"
    answers_path = f"{full_path}.ans"

    # dictionary for questions:
    q_dict = {}
    # examples we will write to our json file
    examples = []

    # Opening up the questions path
    with open(questions_path, newline="", encoding="utf-8") as infile:
        # Reading in .tsv file
        read_file = csv.reader(infile, delimiter="\t")
        for i, row in enumerate(read_file):
            # Declaring arrays for questions, and questions after being organized
            questions_set = []
            organized_questions = []
            # Assigning variables to represent different parts of each row of question data
            q_id = row[0]  # question ID
            story = row[2]
            questions = row[3:]

            # Iterating through the questions:
            for q in questions:
                # stripping some trailing whitespace
                q.lstrip(" ")
                # In the repo, they mentioned the following part is hard-coded!
                if q[0] in ["1", "2", "3", "4"]:
                    # appending q to q_set
                    questions_set = []
                    questions_set.append(q)

                else:
                    if q.startswith("*"):
                        # Removing the asterisk that flags correct answer
                        q = q[:1]
                    questions_set.append(q)

                if len(questions_set) == 5:
                    organized_questions.append(questions_set)

            # running testing that our current organized q has a length of 4
            try:
                assert len(organized_questions) == 4

            except Exception as e:
                print(e)
                print(organized_questions)
                print(len(organized_questions))

            # Not sure if this is needed:
            # assert len(organized_questions) == 4

            # Iterating through the rows of organized questions
            for index in organized_questions:
                assert len(index) == 5  # 1 question, 4 choices for answers

            # Adding story and organized questions to the q_id in q_dict
            q_dict[q_id] = {"story": story, "questions": organized_questions}

    # Going through answers pathway
    with open(answers_path, newline="") as answer_file:
        read_file = csv.reader(answer_file, delimiter="\t")
        for i, row in enumerate(read_file):
            # Getting the id of the story
            id = f"Story:mc160.dev.{i}"
            q_dict[q_id].update({"answers": id})

        # Declaring counter variables for story and question
        story_counter = 0
        question_counter = 0

        # Going through the mc_dict
        for dict_index, mc_dict in q_dict.items():  # q_dict is a nested dict!
            story_counter += 1
            story = mc_dict["story"]
            j = 1  # captures question id later when writing!

            # Going through questions and answers:
            for q, a in zip(mc_dict["questions"], mc_dict["answers"]):
                # map of answer choices:vals
                lut_map = {"A": 1, "B": 2, "C": 3, "D": 4}
                question_choices = q[1:]
                answer_indexed = lut_map[a]
                text_answer = question_choices[answer_indexed]

                question_text = q[0]

                # Doing some cleanup on the text:
                # removing the beginning number
                question_text = question_text[1:]
                # removing the trailing whitespace
                question_text = question_text.lstrip(" ")
                # removing :
                question_text = question_text.lstrip(":")
                # removing the trailing whitespace
                question_text = question_text.lstrip(" ")

                # Cleaning text further, but also getting question type: one vs. multiple
                # Checking for each question type first:
                if question_text.startswith("one"):
                    cleaned = (question_text).strip("one")
                    cleaned = cleaned.lstrip(" ")
                    cleaned = cleaned.lstrip(":")
                    cleaned = cleaned.lstrip(" ")
                    q_type = "one"

                elif question_text.startswith("youn"):
                    cleaned = (question_text).strip("youn")
                    cleaned = cleaned.lstrip(" ")
                    cleaned = cleaned.lstrip(":")
                    cleaned = cleaned.lstrip(" ")
                    q_type = "youn"

                elif question_text.startswith("multipl"):
                    cleaned = (question_text).strip("multipl")
                    cleaned = cleaned.lstrip(" ")
                    cleaned = cleaned.lstrip(":")
                    cleaned = cleaned.lstrip(" ")
                    q_type = "multipl"

                else:
                    cleaned = (question_text).strip("multiple")
                    cleaned = cleaned.lstrip(" ")
                    cleaned = cleaned.lstrip(":")
                    cleaned = cleaned.lstrip(" ")
                    q_type = "multiple"

                # assign the data collected and cleaned as the values for a dictionary:
                e_dict = {
                    "story": story,
                    "story_id": str(dict_index),
                    "question_id": str(j),
                    "question": cleaned,
                    "q_type": q_type,
                    "choices": question_choices,
                    "answer": a,  # as in A, B, C, or D
                    "text_answer": text_answer,
                    "label": str(answer_indexed),
                }

                # Appending e_dict to examples, and upping counters
                examples.append(e_dict)
                j += 1
                question_counter += 1

    # Printing out our results to check!
    print(f"{story_counter} stories")
    print(f"{question_counter} questions")

    # Writing out our final json results!
    with open(f"{path}/{split}", "w", encoding="utf-8") as out_file:
        # Iterating through the examples
        for ex in examples:
            json.dump(ex, out_file)
            out_file.write("\n")


# Enter in files to process below
if __name__ == "__main__":

    # writing the txt files to tsv
    # NOTE remove ablsolute paths for github push
    # NOTE 2 - for output path in txt to tsv, we don't need the full file path
    # txt_to_tsv(file_path=r'example directory!', output_path=r'CreoleTranslations/mc160.dev.kreyol1.tsv')
    # txt_to_tsv(file_path=r'example!!', output_path=r'CreoleTranslations/mc160.dev.kreyol2_localized.tsv')

    # Writing txts to jsons:
    write_to_json(
        path=r"/home/bglid//uni_ms/ling_545/haitian-creole-nlu/models/Data/CreoleTranslations/",
        split=r"mc160.dev.kreyol1",
    )
