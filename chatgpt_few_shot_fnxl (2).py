import openai
import json
import pandas as pd
import csv

openai.api_key = 'mdj3dky-q8Zyd3dltRYRXgYOLm3vmADT3jnejkFJrdedeuH3V1U4l4VDrkJzgHmSGbceuceu'


def chatgpt_response(data):
    outputs = []

    for i in range(len(data)):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",
                 "content": "Based on the instruction and financial statement, answer the question without retries. "
                            "Following are examples, 1: Instruction: First, read the task description. There could be "
                            "multiple numerals reported in a financial statement. Each number may or may not be "
                            "associated with a particular tag. Now read the following financial statement: 'We expect "
                            "to contribute $30 to our defined benefit pension plans in 2021.' What is the tag "
                            "associated with the numeral 30 ? Answer: 'defined benefit plan expected future employer "
                            "contributions next fiscal year', 2: Instruction: First, read the task description. There "
                            "could be multiple numerals reported in a financial statement. Each number may or may not "
                            "be associated with a particular tag. Now read the following financial statement: "
                            "'Accounts receivable, net of allowances of $114 million and $92 million as of December "
                            "31, 2020 and 2019, respectively' What is the tag associated with the numeral 114 ? "
                            "Answer: 'allowance for doubtful accounts receivable current', 3: Instruction: First, "
                            "read the task description. There could be multiple numerals reported in a financial "
                            "statement. Each number may or may not be associated with a particular tag. Now read the "
                            "following financial statement: 'Accounts receivable, net of allowances of $114 million "
                            "and $92 million as of December 31, 2020 and 2019, respectively' What is the tag "
                            "associated with the numeral 92 ? Answer: 'allowance for doubtful accounts receivable "
                            "current', 4: Instruction: First, read the task description. There could be multiple "
                            "numerals reported in a financial statement. Each number may or may not be associated "
                            "with a particular tag. Now read the following financial statement: 'We incurred "
                            "advertising expenses of $1.57 billion, $1.10 billion, and $324 million for the years "
                            "ended December31, 2019, 2018, and 2017, respectively.' What is the tag associated with "
                            "the numeral 1.1 ? Answer: 'advertising expense', 5: Instruction: First, read the task "
                            "description. There could be multiple numerals reported in a financial statement. Each "
                            "number may or may not be associated with a particular tag. Now read the following "
                            "financial statement: 'We incurred advertising expenses of $1.57 billion, $1.10 billion, "
                            "and $324 million for the years ended December31, 2019, 2018, and 2017, respectively.' "
                            "What is the tag associated with the numeral 324 ? Answer: 'advertising expense'. Now "
                            "answer" +
                            data[i]
                 },
            ]
        )

        outputs.append(response['choices'][0]['message']['content'])

    return (outputs)


if __name__ == "__main__":

    openai.api_key = 'mky-q8ZyltRYRXgYOLm3vmADT3jnejkFJrH3V1U4l4VDrkJzgHmSGbceuceu'
    df = pd.read_csv('xbrl_fnxc_data.csv')

    examples = []  # Performing analysis on 500 examples
    for i in range(500):
        examples.append(
            "Instruction: First, read the task description. There could be multiple numerals reported in a financial "
            "statement. Each number may or may not be associated with a particular tag. Now read the following "
            "financial statement. " +
            df['Sentence'][i] + " What is the tag associated with the numeral " + df['Numeral'][i] + " ?")

    chatgpt_response(examples)
