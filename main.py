import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, LSTM, Dropout, GRU, Conv1D, MaxPool1D
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Recall, AUC, Precision
from sklearn.utils.class_weight import compute_class_weight
import pickle
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('data/input_bcell.csv')

df1 = pd.read_csv('data/input_covid.csv')

df2 = pd.read_csv('data/input_sars.csv')



df1_test = df1[['peptide_seq']].copy()

# tokenizer = Tokenizer()
# print(df.head())

# print(df.shape)

# print(df.info())

# print(df.shape)
# print(sum(df['target']==0))
# print(sum(df['target']==1))

# print(df2.shape)
# print(sum(df2['target']==0))
# print(sum(df['target']==1))

combined_df = pd.concat([df,df2],axis=0).reset_index(drop=True).sample(frac=1,random_state=42)



# majority_class = combined_df[combined_df['target']==0]
# minority_class = combined_df[combined_df['target']==1]

# # print(majority_class.shape)
# # print(minority_class.shape)


# minor_length_balance = len(minority_class)*2
# majority_downsampled = majority_class.sample(n=minor_length_balance,random_state=42)

# df_balanced = pd.concat([majority_downsampled,minority_class]).sample(frac=1,random_state=42).reset_index(drop=True)

# print(combined_df.head())

x_df = list(combined_df['peptide_seq'])
x_test_df = list(df1['peptide_seq'])
y_df = combined_df['target']

classes = np.unique(y_df)

class_weight_array = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_df
)

class_weight = dict(zip(classes,class_weight_array))



# tokenizer = Tokenizer(char_level=True)

with open('tokenizer.pkl','rb') as f:
    tokenizer = pickle.load(f)

# tokenizer.fit_on_texts(x_df)


index = tokenizer.word_index

# print(index)

sequences = tokenizer.texts_to_sequences(x_df)
test_sequences = tokenizer.texts_to_sequences(x_test_df)

# print(sequences)

max_len = max(len(seq) for seq in sequences)

padded_sequences = pad_sequences(sequences,maxlen=max_len,padding='post')

padded_test_sequences = pad_sequences(test_sequences,maxlen=max_len,padding='post')

# print(padded_sequences)

vocabsize = len(index)+1


# with open('tokenizer.pkl','wb') as file:
#     pickle.dump(tokenizer,file)

model = Sequential()
model.add(Embedding(input_dim=vocabsize,output_dim=16,input_length=max_len))
model.add(Conv1D(kernel_size=7,filters=32))
model.add(MaxPool1D(pool_size=2,padding='valid'))
model.add(Bidirectional(GRU(units=128)))
model.add(Dropout(0.4))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


recall = Recall()
auc = AUC()
precision = Precision()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=[recall,'accuracy',auc,precision])




earlystop = EarlyStopping(verbose=1,restore_best_weights=True,monitor='val_auc',patience=7,mode='max')

history = model.fit(
    padded_sequences,
    y_df,
    epochs=15,
    batch_size = 32,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks =[earlystop]
)

model.save('bcell_model.keras')

# model = load_model('bcell_model.keras')                                                            


predictions = model.predict(padded_test_sequences)

# print(predictions)



binary_preds = (predictions>0.5).astype(int)

print(binary_preds)

df1_test['predictions'] = predictions.flatten()
df1_test['binary_predictions'] = binary_preds

# print(df1_test)

df1_predicted_sorted = df1_test.sort_values(by='predictions',ascending=False)

print(df1_predicted_sorted)
print(class_weight)

df1_predicted_sorted.to_csv('covid_predicted_epitopes.csv',index=False)            