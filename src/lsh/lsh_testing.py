import numpy as np
import pandas as pd
import time
import lsh


def embeddings_to_dataframe(file_path):
    df = pd.read_csv(file_path, sep=' ', header=None)
    return df

def evaluate_lsh(lsh, test_real_embeddings, test_fake_embeddings,real_embeddings_df, fake_embeddings_df):
    correct_real = 0
    total_real = len(test_real_embeddings)
    total_real_time = 0  
    
    correct_fake = 0
    total_fake = len(test_fake_embeddings)
    total_fake_time = 0 

    start_time = time.time()
    # Insert embeddings into the LSH
    for _, real_embedding in real_embeddings_df.iterrows():
        lsh.insert(real_embedding.values.astype(np.float32), real=True)

    for _, fake_embedding in fake_embeddings_df.iterrows():
        lsh.insert(fake_embedding.values.astype(np.float32), real=False)
    end_time = time.time()

    load_time = (end_time - start_time) * 1000


    # Evaluate real embeddings
    for _, test_embedding in test_real_embeddings.iterrows():
        test_embedding = test_embedding.values.astype(np.float32)
        
        start_time = time.time()  
        neighbors = lsh.query(test_embedding)
        

        real_score = 0
        fake_score = 0
        
        for _, label, distance in neighbors:
            # Inverse distance
            weight = 1 / (.000001 + distance) 
            if label == 'real':
                real_score += weight
            else:
                fake_score += weight
        
        predicted_label = 'real' if real_score > fake_score else 'fake'
        if predicted_label == 'real':
            correct_real += 1

        end_time = time.time()  
        
        total_real_time += (end_time - start_time) * 1000  

    accuracy_real = correct_real / total_real

    # Evaluate fake embeddings
    for _, test_embedding in test_fake_embeddings.iterrows():
        test_embedding = test_embedding.values.astype(np.float32)
        
        start_time = time.time()  
        neighbors = lsh.query(test_embedding)
        

        real_score = 0
        fake_score = 0
        
        for _, label, distance in neighbors:
            weight = 1 / (1 + distance)
            if label == 'real':
                real_score += weight
            else:
                fake_score += weight
        
        predicted_label = 'real' if real_score > fake_score else 'fake'
        
        if predicted_label == 'fake':
            correct_fake += 1

        end_time = time.time()
        
        total_fake_time += (end_time - start_time) * 1000

    accuracy_fake = correct_fake / total_fake
    

    avg_time = (total_fake_time + total_real_time + load_time)/(total_fake + total_real)


    return accuracy_real, accuracy_fake, avg_time