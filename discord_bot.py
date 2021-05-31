"""
author: Alex Sutay
file: betbot.py
"""

import discord
import re
from config import TOKEN
import numpy as np
from threading import Lock

client = discord.Client()
global vocab
global vocab_lock
SAVE_FILE = 'discord_data.txt'


async def read_channel(message):
    await message.channel.send("Reading....")
    messages = await message.channel.history(limit=50000).flatten()
    await message.channel.send(str(len(messages)))
    await message.channel.send("Analyzing...")

    master_str = ''
    words = {}
    for msg in messages:
        """
        if not msg.author.bot:
            regex = re.compile('[^a-zA-Z ]')
            content = regex.sub('', msg.content)
            content = content.lower()
            master_str = content + ' ' + master_str
            split_data = content.split(' ')
            for word in split_data:
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
                    """
        master_str = msg.content + '\0' + master_str

    print(master_str)
    with open(SAVE_FILE, 'a', encoding='utf-8') as f:
        f.write(master_str)

    await message.channel.send("All done! Now I just gotta figure out what to say...")

    """sorted_words = {k: v for k, v in sorted(words.items(), key=lambda item: item[1])}
    print(sorted_words)
    await message.channel.send("Top words: " + str(sorted_words)[-1500:])
    "
    count = 1
    vocabulary = {'<OTHER>': 1}
    for word in sorted_words:
        if sorted_words[word] >= 5:
            count += 1
            vocabulary[word] = count

    print(vocabulary)
    print(len(vocabulary))
    await save_vocab(vocabulary)

    master_list = []
    master_str = regex.sub('', master_str)
    master_str = master_str.split(' ')
    for word in master_str:
        if word in vocabulary:
            master_list.append(vocabulary[word])
        else:
            master_list.append(1)

    print(master_list)

    outmatrix = None
    for row in range(len(master_str) - 20):
        this_array = master_list[row:row + 20]
        if outmatrix is None:
            outmatrix = this_array
        elif np.size(outmatrix) == 20:
            outmatrix = np.asarray([outmatrix, this_array])
        else:
            outmatrix = np.vstack([outmatrix, this_array])

    print(outmatrix)

    outdict = {'X': outmatrix.astype(float)}
    sio.savemat('Out.mat', outdict)
    print('done')
    """


@client.event
async def on_ready():
    print("Logged in as")
    print(client.user.name)
    print(client.user.id)


async def wisdom(msg):
    """
    print out a response 19 words long
    :param msg:
    :return:
    global vocab_lock
    vocab_lock.acquire()
    global vocab
    rev_vocab = {v: k for k, v in vocab.items()}
    messages = await msg.channel.history(limit=30).flatten()
    idx1 = 0
    idx2 = 0
    word_lst = []
    while len(word_lst) < 19:
        if idx2 == 0:
            this_msg = messages[idx1]
            idx1 += 1
            while this_msg.author.bot:
                this_msg = messages[idx1]
                idx1 += 1
            this_msg = this_msg.content.split(' ')
            idx2 = len(this_msg)
        else:
            idx2 -= 1
            if this_msg[idx2] in vocab:  # todo apply the regex to these strings
                word_lst.append(vocab[this_msg[idx2]])

    thetas = predict_next.thetas_from_mat(THETA_FILE)
    final_msg = ''
    for i in range(19):
        next_word = predict_next.predict(thetas, np.asarray(word_lst))
        print(next_word)
        next_word = np.argmax(next_word) + 1
        final_msg += rev_vocab[next_word] + ' '
        word_lst.pop(0)
        word_lst.append(next_word)

    await msg.channel.send(final_msg)
    vocab_lock.release()
    """


CMDs = {
    '!read_channel' : read_channel,
    '!wisdom' : wisdom,
}


@client.event
async def on_message(message):
    if message.content in CMDs:
        await CMDs[message.content](message)


if __name__ == '__main__':
    vocab_lock = Lock()
    # load_vocab()
    client.run(TOKEN)
