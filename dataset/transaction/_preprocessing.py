import pickle

# context = pickle.load(open('./order_table/order_context.pkl', 'rb'))
# print(context)
#
# item = pickle.load(open('./order_table/order_item.pkl', 'rb'))
# print(item)
#
# domain = pickle.load(open('./order_table/order_domain.pkl', 'rb'))
# print(domain)
#
# user = pickle.load(open('./order_table/order_user.pkl', 'rb'))
# print(user)

transaction = pickle.load(open('../order_table/transaction.pkl', 'rb'))  # user(1)-context(3)-domain(1)-item(15)
print('len(transaction)', len(transaction))


print('len(set(transaction))', len(list(set([tuple(item) for item in transaction]))))
transaction = list(set([tuple(item) for item in transaction]))

#
# vocab = []
# for t in transaction:
#     vocab += t
# # print(len(vocab))
#
# vocabs = list(set(vocab) - set(['']))
# # print(len(vocabs))
#
# vocab = {
#     'size': len(vocabs) + 2,
#     'special_tokens': ['<pad>', '<mask>'],
#     'idx_to_vocab': {idx: vocab for idx, vocab in enumerate(['<pad>', '<mask>'] + vocabs)},
#     'vocab_to_idx': {vocab: idx for idx, vocab in enumerate(['<pad>', '<mask>'] + vocabs)},
# }
#
# print('len(vocab)', vocab['size'])
#
# pickle.dump(vocab, open('vocab.pkl', 'wb'))

vocab = pickle.load(open('vocab.pkl', 'rb'))

# cnt = {i: 0 for i in range(1, 16)}
# for t in transaction:
#     for i, v in enumerate(t):
#         if v == '':
#             cnt[i - 5] += 1
#             break
#
# print(cnt)

u_1, u_t = [], []
save_1, save_t = [], []
for t in transaction:
    if t[6] == '':
        save = []
        save_1.append(
            [vocab['vocab_to_idx'][v if v != '' else '<pad>'] for v in t]
        )
        if save_1[-1][0] not in u_1:
            u_1.append(save_1[-1][0])
    else:
        save_t.append(
            [vocab['vocab_to_idx'][v if v != '' else '<pad>'] for v in t]
        )
        if save_t[-1][0] not in u_t:
            u_t.append(save_t[-1][0])

import random
random.shuffle(save_t)
random.shuffle(save_t)
random.shuffle(save_t)
random.shuffle(save_t)

print('len(save_1)', len(save_1))
print('len(save_t)', len(save_t))

print('len(u_1)', len(set(u_1)))
print('len(u_t)', len(set(u_t)))
print('len(u_1) - nlen(u_t)', len(set(u_1) - set(u_t)))
print('len(u_t) - nlen(u_1)', len(set(u_t) - set(u_1)))
print('len(u_1) & nlen(u_t)', len(set(u_1) & set(u_t)))

pretrain_transaction = save_1

# pickle.dump(pretrain_transaction, open("pretrain.pkl", 'wb'))

for i in range(1, 6):
    random.shuffle(save_t)
    random.shuffle(save_t)
    random.shuffle(save_t)
    random.shuffle(save_t)
    random.shuffle(save_t)
    random.shuffle(save_t)
    random.shuffle(save_t)
    random.shuffle(save_t)
    random.shuffle(save_t)
    random.shuffle(save_t)
    random.shuffle(save_t)
    random.shuffle(save_t)
    random.shuffle(save_t)
    random.shuffle(save_t)
    random.shuffle(save_t)
    random.shuffle(save_t)
    random.shuffle(save_t)
    train_transaction = save_t[:int(len(save_t)*0.8)]
    valid_transaction = save_t[int(len(save_t)*0.8):int(len(save_t)*0.9)]
    test_transaction = save_t[int(len(save_t)*0.9):]

    # pickle.dump(train_transaction, open(f"train_{i}.pkl", 'wb'))
    # pickle.dump(valid_transaction, open(f"valid_{i}.pkl", 'wb'))
    # pickle.dump(test_transaction, open(f"test_{i}.pkl", 'wb'))