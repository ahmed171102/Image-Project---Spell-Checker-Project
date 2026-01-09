"""
model.py - Spell Checker Seq2Seq Model Architecture

This file contains:
- Encoder-Decoder model with Attention
- Inference models for prediction
- Beam search decoding function

Usage:
    from model import create_spell_checker_model, beam_search_decode
"""

import numpy as np
import math
import re
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Embedding, 
    Bidirectional, Concatenate, Activation, Dot
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# DEFAULT HYPERPARAMETERS
# ============================================================
DEFAULT_LATENT_DIM = 128
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_MAX_SEQ_LEN = 80
DEFAULT_BEAM_WIDTH = 3


# ============================================================
# ATTENTION MECHANISM
# ============================================================
def attention_layer(query, value):
    """
    Dot-product attention mechanism.
    
    Args:
        query: Decoder output (batch, time_steps, features)
        value: Encoder output (batch, time_steps, features)
    
    Returns:
        Context vector weighted by attention scores
    """
    score = Dot(axes=[2, 2])([query, value])
    attention_weights = Activation('softmax')(score)
    context_vector = Dot(axes=[2, 1])([attention_weights, value])
    return context_vector


# ============================================================
# TRAINING MODEL
# ============================================================
def build_training_model(vocab_size, 
                         latent_dim=DEFAULT_LATENT_DIM, 
                         embedding_dim=DEFAULT_EMBEDDING_DIM, 
                         max_seq_len=DEFAULT_MAX_SEQ_LEN):
    """
    Build the full encoder-decoder model for training.
    
    Args:
        vocab_size: Size of character vocabulary
        latent_dim: LSTM hidden units (default: 128)
        embedding_dim: Character embedding dimensions (default: 64)
        max_seq_len: Maximum sequence length (default: 80)
    
    Returns:
        Dictionary containing:
        - model: Compiled Keras model
        - encoder_inputs: Encoder input layer
        - encoder_outputs: Encoder output tensor
        - encoder_states: [state_h, state_c]
        - decoder_lstm: Decoder LSTM layer
        - dec_emb_layer: Decoder embedding layer
        - decoder_dense: Output dense layer
    """
    
    # ==================== ENCODER ====================
    encoder_inputs = Input(shape=(max_seq_len,), name='encoder_inputs')
    
    # Embedding layer: converts character indices to dense vectors
    enc_emb = Embedding(
        input_dim=vocab_size, 
        output_dim=embedding_dim, 
        input_length=max_seq_len, 
        mask_zero=True,
        name='encoder_embedding'
    )(encoder_inputs)
    
    # Bidirectional LSTM: reads input forward and backward
    encoder_lstm = Bidirectional(
        LSTM(latent_dim, return_sequences=True, return_state=True),
        name='encoder_bidirectional'
    )
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(enc_emb)
    
    # Concatenate forward and backward states
    state_h = Concatenate(name='encoder_state_h')([forward_h, backward_h])
    state_c = Concatenate(name='encoder_state_c')([forward_c, backward_c])
    encoder_states = [state_h, state_c]
    
    # ==================== DECODER ====================
    decoder_inputs = Input(shape=(max_seq_len,), name='decoder_inputs')
    
    # Decoder embedding layer (saved for inference model)
    dec_emb_layer = Embedding(
        input_dim=vocab_size, 
        output_dim=embedding_dim, 
        input_length=max_seq_len, 
        mask_zero=True,
        name='decoder_embedding'
    )
    dec_emb = dec_emb_layer(decoder_inputs)
    
    # Decoder LSTM: generates output sequence
    # Note: latent_dim * 2 because encoder is bidirectional
    decoder_lstm = LSTM(
        latent_dim * 2, 
        return_sequences=True, 
        return_state=True,
        name='decoder_lstm'
    )
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    
    # ==================== ATTENTION ====================
    attention_result = attention_layer(decoder_outputs, encoder_outputs)
    
    # Concatenate decoder output with attention context
    decoder_concat = Concatenate(axis=-1, name='attention_concat')(
        [decoder_outputs, attention_result]
    )
    
    # ==================== OUTPUT ====================
    decoder_dense = Dense(vocab_size, activation='softmax', name='output_dense')
    final_outputs = decoder_dense(decoder_concat)
    
    # ==================== COMPILE ====================
    model = Model(
        inputs=[encoder_inputs, decoder_inputs], 
        outputs=final_outputs,
        name='spell_checker_seq2seq'
    )
    model.compile(
        optimizer='rmsprop', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # Return model and all components needed for inference
    return {
        'model': model,
        'encoder_inputs': encoder_inputs,
        'encoder_outputs': encoder_outputs,
        'encoder_states': encoder_states,
        'encoder_lstm': encoder_lstm,
        'decoder_lstm': decoder_lstm,
        'dec_emb_layer': dec_emb_layer,
        'decoder_dense': decoder_dense
    }


# ============================================================
# INFERENCE MODELS
# ============================================================
def build_inference_models(encoder_inputs, encoder_outputs, encoder_states,
                           decoder_lstm, dec_emb_layer, decoder_dense,
                           latent_dim=DEFAULT_LATENT_DIM,
                           max_seq_len=DEFAULT_MAX_SEQ_LEN):
    """
    Build separate encoder and decoder models for inference.
    
    During inference, we can't use teacher forcing, so we need
    separate models that process one character at a time.
    
    Args:
        encoder_inputs: Encoder input layer from training model
        encoder_outputs: Encoder output tensor
        encoder_states: [state_h, state_c] from encoder
        decoder_lstm: Decoder LSTM layer
        dec_emb_layer: Decoder embedding layer
        decoder_dense: Output dense layer
        latent_dim: LSTM hidden units
        max_seq_len: Maximum sequence length
    
    Returns:
        encoder_model: Encodes input sequence
        decoder_model: Decodes one character at a time
    """
    
    # ==================== ENCODER INFERENCE MODEL ====================
    # Takes input sequence, returns encoder outputs and states
    encoder_model = Model(
        inputs=encoder_inputs, 
        outputs=[encoder_outputs] + encoder_states,
        name='encoder_inference'
    )
    
    # ==================== DECODER INFERENCE MODEL ====================
    # State inputs (from previous step)
    decoder_state_input_h = Input(shape=(latent_dim * 2,), name='decoder_state_h_input')
    decoder_state_input_c = Input(shape=(latent_dim * 2,), name='decoder_state_c_input')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    # Single character input
    decoder_inputs_inf = Input(shape=(1,), name='decoder_single_input')
    dec_emb_inf = dec_emb_layer(decoder_inputs_inf)
    
    # Run one LSTM step
    decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
        dec_emb_inf, initial_state=decoder_states_inputs
    )
    decoder_states_inf = [state_h_inf, state_c_inf]
    
    # Encoder outputs input (for attention)
    encoder_outputs_input = Input(
        shape=(max_seq_len, latent_dim * 2), 
        name='encoder_outputs_input'
    )
    
    # Apply attention
    attention_result_inf = attention_layer(decoder_outputs_inf, encoder_outputs_input)
    decoder_concat_inf = Concatenate(axis=-1)([decoder_outputs_inf, attention_result_inf])
    
    # Get output probabilities
    decoder_outputs_inf = decoder_dense(decoder_concat_inf)
    
    # Build decoder model
    decoder_model = Model(
        inputs=[decoder_inputs_inf, decoder_state_input_h, 
                decoder_state_input_c, encoder_outputs_input],
        outputs=[decoder_outputs_inf] + decoder_states_inf,
        name='decoder_inference'
    )
    
    return encoder_model, decoder_model


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def restore_numbers(original_text, predicted_text):
    """
    Keep numbers from original text stable.
    
    The model sometimes changes numbers (e.g., '3600' -> '3500').
    This function replaces predicted numbers with original ones.
    
    Args:
        original_text: Original input string
        predicted_text: Model's prediction
    
    Returns:
        Prediction with original numbers restored
    """
    input_nums = re.findall(r'\d+', original_text)
    pred_nums = re.findall(r'\d+', predicted_text)
    
    if len(input_nums) > 0 and len(input_nums) == len(pred_nums):
        for i in range(len(input_nums)):
            predicted_text = predicted_text.replace(pred_nums[i], input_nums[i], 1)
    return predicted_text


# ============================================================
# BEAM SEARCH DECODING
# ============================================================
def beam_search_decode(input_text, tokenizer, encoder_model, decoder_model,
                       max_seq_len=DEFAULT_MAX_SEQ_LEN, 
                       beam_width=DEFAULT_BEAM_WIDTH):
    """
    Beam search decoding for spell correction.
    
    Beam search keeps track of multiple candidate sequences
    and selects the most probable one at the end.
    
    Args:
        input_text: Noisy input string to correct
        tokenizer: Fitted Keras Tokenizer (character-level)
        encoder_model: Encoder inference model
        decoder_model: Decoder inference model
        max_seq_len: Maximum sequence length (default: 80)
        beam_width: Number of beams/candidates to keep (default: 3)
    
    Returns:
        Corrected text string
    """
    
    # Prepare input sequence
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_pad = pad_sequences(input_seq, maxlen=max_seq_len, padding='post')
    
    # Encode input
    enc_outs, h, c = encoder_model.predict(input_pad, verbose=0)
    
    # Build reverse character map for decoding
    reverse_char_map = {i: char for char, i in tokenizer.word_index.items()}
    
    # Get start token index
    start_token = tokenizer.word_index['\t']
    
    # Initialize beams: [score, sequence, last_token, state_h, state_c]
    beams = [[0.0, "", start_token, h, c]]
    
    # Decode character by character
    for _ in range(max_seq_len):
        all_candidates = []
        
        for beam in beams:
            score, seq, last_tok, st_h, st_c = beam
            
            # Check if sequence has ended
            if len(seq) > 0 and seq[-1] == '\n':
                all_candidates.append(beam)
                continue
            
            # Prepare input for decoder
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = last_tok
            
            # Predict next character probabilities
            probs, new_h, new_c = decoder_model.predict(
                [target_seq, st_h, st_c, enc_outs], verbose=0
            )
            probs = probs[0, -1, :]
            
            # Get top-k candidate characters
            top_indices = np.argsort(probs)[-beam_width:]
            
            for idx in top_indices:
                p = probs[idx]
                if p < 1e-10:
                    continue
                    
                # Calculate new score (log probability)
                new_score = score + math.log(p)
                char = reverse_char_map.get(idx, '')
                new_seq = seq + char
                all_candidates.append([new_score, new_seq, idx, new_h, new_c])
        
        # Keep only top beams
        beams = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:beam_width]
        
        # Early stop if all beams have ended
        if all(b[1].endswith('\n') for b in beams):
            break
    
    # Return best sequence
    best_seq = beams[0][1].strip()
    return restore_numbers(input_text, best_seq)


# ============================================================
# CONVENIENCE FUNCTION - CREATE ALL MODELS AT ONCE
# ============================================================
def create_spell_checker_model(vocab_size, 
                                latent_dim=DEFAULT_LATENT_DIM,
                                embedding_dim=DEFAULT_EMBEDDING_DIM,
                                max_seq_len=DEFAULT_MAX_SEQ_LEN):
    """
    Create both training and inference models in one call.
    
    This is the easiest way to set up the complete model.
    
    Args:
        vocab_size: Size of character vocabulary
        latent_dim: LSTM hidden units (default: 128)
        embedding_dim: Embedding dimensions (default: 64)
        max_seq_len: Maximum sequence length (default: 80)
    
    Returns:
        Dictionary containing:
        - training_model: For model.fit()
        - encoder_model: For inference encoding
        - decoder_model: For inference decoding
    
    Example:
        models = create_spell_checker_model(vocab_size=100)
        models['training_model'].fit(...)
        result = beam_search_decode(text, tokenizer, 
                                   models['encoder_model'],
                                   models['decoder_model'])
    """
    
    # Build training model and get components
    components = build_training_model(
        vocab_size=vocab_size, 
        latent_dim=latent_dim, 
        embedding_dim=embedding_dim, 
        max_seq_len=max_seq_len
    )
    
    # Build inference models using training model components
    encoder_model, decoder_model = build_inference_models(
        encoder_inputs=components['encoder_inputs'],
        encoder_outputs=components['encoder_outputs'],
        encoder_states=components['encoder_states'],
        decoder_lstm=components['decoder_lstm'],
        dec_emb_layer=components['dec_emb_layer'],
        decoder_dense=components['decoder_dense'],
        latent_dim=latent_dim,
        max_seq_len=max_seq_len
    )
    
    return {
        'training_model': components['model'],
        'encoder_model': encoder_model,
        'decoder_model': decoder_model
    }


# ============================================================
# LOAD TRAINED MODEL FOR INFERENCE
# ============================================================
def load_model_for_inference(model_path, vocab_size,
                              latent_dim=DEFAULT_LATENT_DIM,
                              embedding_dim=DEFAULT_EMBEDDING_DIM,
                              max_seq_len=DEFAULT_MAX_SEQ_LEN):
    """
    Load a trained model and prepare it for inference.
    
    Args:
        model_path: Path to saved .h5 model file
        vocab_size: Vocabulary size used during training
        latent_dim: LSTM units used during training
        embedding_dim: Embedding size used during training
        max_seq_len: Max sequence length used during training
    
    Returns:
        encoder_model, decoder_model ready for beam_search_decode()
    """
    
    # Create model architecture
    models = create_spell_checker_model(
        vocab_size=vocab_size,
        latent_dim=latent_dim,
        embedding_dim=embedding_dim,
        max_seq_len=max_seq_len
    )
    
    # Load trained weights
    models['training_model'].load_weights(model_path)
    
    return models['encoder_model'], models['decoder_model']


# ============================================================
# TEST (runs only when executed directly)
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing model.py")
    print("=" * 60)
    
    # Test with example vocabulary size
    VOCAB_SIZE = 100
    
    print(f"\n1. Creating models with vocab_size={VOCAB_SIZE}...")
    models = create_spell_checker_model(VOCAB_SIZE)
    
    print("\n2. Training model summary:")
    print("-" * 40)
    models['training_model'].summary()
    
    print("\n3. Model shapes:")
    print(f"   - Encoder input:  {models['encoder_model'].input_shape}")
    print(f"   - Encoder output: {models['encoder_model'].output_shape}")
    print(f"   - Decoder input:  {models['decoder_model'].input_shape}")
    print(f"   - Decoder output: {models['decoder_model'].output_shape}")
    
    print("\n" + "=" * 60)
    print("✅ model.py is working correctly!")
    print("=" * 60)
    print("\nUsage:")
    print("  from model import create_spell_checker_model, beam_search_decode")
    print("  models = create_spell_checker_model(vocab_size)")
    print("  models['training_model'].fit(...)")
