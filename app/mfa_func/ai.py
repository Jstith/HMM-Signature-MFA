import numpy as np
import pickle
import base64
from hmmlearn.hmm import GaussianHMM


def train_model(user_id, inp_data):

    np_data = np.array(inp_data)
    num_sequences, sequence_length, num_features = np_data.shape
    X = np_data.reshape(-1, 3)
    lengths = [len(inp_data[0]), len(inp_data[1]), len(inp_data[2])]

    n_states = 6
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=1337)
    model.fit(X, lengths)

    train_scores = [model.score(seq) for seq in inp_data]
    mean_score = np.mean(train_scores)
    std_score = np.std(train_scores)
    threshold = mean_score - 1.3 * std_score

    model_bytes = pickle.dumps(model)
    model_b64 = base64.b64encode(model_bytes).decode('utf-8')
    print('Generated model base64:')
    print(model_b64)

    return model_b64, threshold

def query_user(inp_model_b64, inp_threshold, inp_seq_data):
    model_bytes = base64.b64decode(inp_model_b64)
    model = pickle.loads(model_bytes)
    np_data = np.array(inp_seq_data)
    print(f"shape of data getting shoved into model is {np_data.shape}")
    log_likelihood = model.score(np_data)

    print(f'[+] Input likelihook {log_likelihood}, threshold likelihood {inp_threshold}')
    if(log_likelihood > inp_threshold):
        return True
    return False
