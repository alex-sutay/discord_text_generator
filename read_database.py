import sqlite3


con = sqlite3.connect("data/database.sqlite")
cur = con.cursor()

cur.execute('SELECT * FROM quotes')
records = cur.fetchall()

data = {}

for entry in records:
    row_num, phrase = entry
    if len(phrase.split(' ')) > 9:
        data[row_num] = phrase

words = {}
for row in data:
    split_data = data[row].split(' ')
    for word in split_data:
        if word in words:
            words[word] += 1
        else:
            words[word] = 1

print({k: v for k, v in sorted(words.items(), key=lambda item: item[1])})

# print(words)
print(len(words))

