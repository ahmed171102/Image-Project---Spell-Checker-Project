import numpy as np
import tensorflow as tf
import pickle
import re
import math
from flask import Flask, request, jsonify, render_template
from spellchecker import SpellChecker

# --- TF 1.x Setup ---
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1.keras.models import load_model, Model
from tensorflow.compat.v1.keras.layers import Input, Dot, Activation, Concatenate
from tensorflow.compat.v1.keras.preprocessing.sequence import pad_sequences
from tensorflow.compat.v1.keras import backend as K

# Initialize Flask App
app = Flask(__name__)

# Config
MAX_SEQ_LEN = 80
LATENT_DIM = 256
BEAM_WIDTH = 3

# Global Variables
model = None
encoder_model_inf = None
decoder_model_inf = None
tokenizer = None
reverse_char_map = None
target_token_index = None
spell = SpellChecker()
sess = tf.Session()
graph = tf.get_default_graph()

def manual_attention(query, value):
    """Re-creates attention mechanism for inference"""
    score = Dot(axes=[2, 2])([query, value])
    attention_weights = Activation('softmax')(score)
    context_vector = Dot(axes=[2, 1])([attention_weights, value])
    return context_vector

def load_resources():
    global model, encoder_model_inf, decoder_model_inf, tokenizer, reverse_char_map, target_token_index
    
    # 1. Load Tokenizer
    print("Loading tokenizer...")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    reverse_char_map = {i: char for char, i in tokenizer.word_index.items()}
    target_token_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1

    # 2. Load Trained Model
    print("Loading model...")
    K.set_session(sess)
    with graph.as_default():
       
        model = load_model('best_speller_pro.h5', compile=False)

        
        encoder_inputs = model.input[0] # Input 1
        encoder_outputs, f_h, f_c, b_h, b_c = model.get_layer('bidirectional').output
        state_h = Concatenate()([f_h, b_h])
        state_c = Concatenate()([f_c, b_c])
        encoder_states = [state_h, state_c]
        
        # Build Encoder Inference Model
        encoder_model_inf = Model(encoder_inputs, [encoder_outputs] + encoder_states)

        # Build Decoder Inference Model
        decoder_lstm = model.get_layer('lstm_1')
        decoder_dense = model.get_layer('dense')
        dec_emb_layer = model.get_layer('embedding_1')

        decoder_state_input_h = Input(shape=(LATENT_DIM * 2,))
        decoder_state_input_c = Input(shape=(LATENT_DIM * 2,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_inputs_inf = Input(shape=(1,))
        dec_emb_inf = dec_emb_layer(decoder_inputs_inf)

        decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(dec_emb_inf, initial_state=decoder_states_inputs)
        decoder_states_inf = [state_h_inf, state_c_inf]

        # Re-apply Attention
        encoder_outputs_input = Input(shape=(MAX_SEQ_LEN, LATENT_DIM * 2))
        attention_result_inf = manual_attention(decoder_outputs_inf, encoder_outputs_input)
        decoder_concat_inf = Concatenate(axis=-1)([decoder_outputs_inf, attention_result_inf])
        decoder_outputs_inf = decoder_dense(decoder_concat_inf)

        decoder_model_inf = Model(
            [decoder_inputs_inf, decoder_state_input_h, decoder_state_input_c, encoder_outputs_input],
            [decoder_outputs_inf] + decoder_states_inf
        )
        print("✅ Models Reconstructed Successfully")

# --- Inference Logic (Same as your script) ---
def restore_numbers(original_text, pred_text):
    input_nums = re.findall(r'\d+', original_text)
    pred_nums = re.findall(r'\d+', pred_text)
    if len(input_nums) > 0 and len(input_nums) == len(pred_nums):
        for i in range(len(input_nums)):
            pred_text = pred_text.replace(pred_nums[i], input_nums[i], 1)
    return pred_text

def apply_spellcheck_filter(text):
    final_words = []
    for word in text.split():
        if word.isdigit() or len(word) < 4:
            final_words.append(word)
            continue
        if word not in spell:
            correction = spell.correction(word)
            final_words.append(correction if correction else word)
        else:
            final_words.append(word)
    return " ".join(final_words)

def beam_search_decode(input_text, k=BEAM_WIDTH):
    K.set_session(sess)
    with graph.as_default():
        input_seq = tokenizer.texts_to_sequences([input_text])
        input_pad = pad_sequences(input_seq, maxlen=MAX_SEQ_LEN, padding='post')
        
        enc_outs, h, c = encoder_model_inf.predict(input_pad)
        
        start_token = target_token_index['\t']
        beams = [[0.0, "", start_token, h, c]]
        
        for _ in range(MAX_SEQ_LEN):
            all_candidates = []
            for b in beams:
                score, seq, last_tok, st_h, st_c = b
                if len(seq) > 0 and seq[-1] == '\n':
                    all_candidates.append(b)
                    continue
                
                target_seq = np.zeros((1, 1))
                target_seq[0, 0] = last_tok
                
                probs, new_h, new_c = decoder_model_inf.predict([target_seq, st_h, st_c, enc_outs])
                probs = probs[0, -1, :]
                top_indices = np.argsort(probs)[-k:]
                
                for idx in top_indices:
                    p = probs[idx]
                    if p < 1e-10: continue
                    new_score = score + math.log(p)
                    char = reverse_char_map.get(idx, '')
                    new_seq = seq + char
                    all_candidates.append([new_score, new_seq, idx, new_h, new_c])
            
            beams = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:k]
            if all(b[1].endswith('\n') for b in beams):
                break
                
        best_seq = beams[0][1].strip()
        best_seq = restore_numbers(input_text, best_seq)
        final_seq = apply_spellcheck_filter(best_seq)
        return final_seq

# Load everything on startup
load_resources()

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    corrected = beam_search_decode(text)
    return jsonify({'original': text, 'corrected': corrected})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)