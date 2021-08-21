"""
author: Alex Sutay
file: betbot.py
"""

# pip libraries
import discord
# python libraries
import os
from threading import Lock
# my libraries
from config import TOKEN
import train

client = discord.Client()
global vocab
global vocab_lock
global MODEL
SAVE_FILE = 'discord_data.txt'  # where the training data is saved
MAX_CHANNELS = 1  # how many channels are allowed in each discord server


async def read_channel(message):
    await message.channel.send("Reading....")
    messages = await message.channel.history(limit=50000).flatten()
    await message.channel.send(str(len(messages)))
    await message.channel.send("Analyzing...")

    master_str = ''
    for msg in messages:
        master_str = msg.content + b'\x00' + master_str

    print(master_str)
    with open(SAVE_FILE, 'a', encoding='utf-8') as f:
        f.write(master_str)

    await message.channel.send("All done! Now I just gotta figure out what to say...")


@client.event
async def on_ready():
    print("Logged in as")
    print(client.user.name)
    print(client.user.id)


async def wisdom(msg):
    master_str = b'\x00'
    messages = await msg.channel.history(limit=10).flatten()
    for msg in messages:
        if not msg.author.bot:
            master_str = msg.content.encode('utf-8') + b'\x00' + master_str
    response = train.generate_one_message(MODEL, master_str)[:-1]  # Chop the last character, it's a null
    await msg.channel.send(response)


def load_model():
    global MODEL
    dataset, ids_from_chars, chars_from_ids = train.get_data('discord_data.txt')
    model = train.create_model(ids_from_chars)
    train.restore(model, 20, os.path.join('./training_checkpoints_discord_2', "ckpt_{epoch}.ckpt"))
    MODEL = train.OneStep(model, chars_from_ids, ids_from_chars)


CMDs = {
    '!read_channel' : read_channel,
    '!wisdom' : wisdom,
}


@client.event
async def on_message(message):
    if message.content in CMDs:
        await CMDs[message.content](message)
    if client.user in message.mentions:
        await wisdom(message)  # if the bot is @-ed, respond


if __name__ == '__main__':
    vocab_lock = Lock()
    # load_vocab()
    print('loading model...')
    load_model()
    print('loaded! logging in')
    client.run(TOKEN)
