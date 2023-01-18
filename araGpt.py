from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments


import torch

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f'Running on GPU {device}')
else:
    print("Running on CPU on")




def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset

def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset


def load_data_collator(tokenizer, mlm=False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
    )
    return data_collator


def train(train_file_path, model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)

    model = GPT2LMHeadModel.from_pretrained(model_name)

    model.save_pretrained(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()


# # you need to set parameters
# train_file_path = "araPoem.txt"
# model_name = 'mabaji/thepoet'
output_dir = '../resultarPoemALL2'
# overwrite_output_dir = False
# per_device_train_batch_size = 4
# num_train_epochs = 10
# save_steps = 1000
# It takes about 30 minutes to train in colab.
# train(
#     train_file_path=train_file_path,
#     model_name=model_name,
#     output_dir=output_dir,
#     overwrite_output_dir=overwrite_output_dir,
#     per_device_train_batch_size=per_device_train_batch_size,
#     num_train_epochs=num_train_epochs,
#     save_steps=save_steps
# )
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer


def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(output_dir)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def generate_text(sequence, max_length):
    model_path = output_dir
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    return  tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    # print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))


# while 1==1:
#         print("Ask me!")
#         sequence = input()  # oil price
#         print("Answer Max Length =  ? ")
#         max_len = int(input())  # 20
#         c = generate_text(sequence,
#                           max_len)  # oil price for July June which had been low at as low as was originally stated Prices have since resumed
#
#         print(c)

from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# app.py
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# app.py
# app.py
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", generated_text='')

@app.route("/generate_poem", methods=["POST"])
def generate_poem():
    user_input = request.form["poem-input"]

    generated_text = generate_text(user_input,300)
    lines = generated_text.split(".")

    print(lines)
    res = ''
    # for line in lines:
    #     res = res + line +'<br/>'
    return render_template("index.html", generated_text= lines)

# @app.route("/generate_poem", methods=["POST"])
# def generate_poem():
#         user_input = request.form["poem-input"]
#         generated_text = generate_text(user_input,50)
#         lines = generated_text.split(".")
#
#         res = ''
#         for line in lines:
#             res = res + '<br>' + line +'</br>'
#         return jsonify(res)

if __name__ == '__main__':
    app.run(debug=True)


# def generate_text(sequence, max_length=100):
#     model = GPT2LMHeadModel.from_pretrained("../resultarPoemALL2")
#     tokenizer = GPT2Tokenizer.from_pretrained("../resultarPoemALL2")
#     input_ids = tokenizer.encode(sequence, return_tensors="pt")
#     generated_text = model.generate(input_ids, max_length=max_length)
#     if len(generated_text) > 0:
#         return generated_text
#     else:
#       return "Nothing generated"
#
# if __name__ == "__main__":
#     app.run()

