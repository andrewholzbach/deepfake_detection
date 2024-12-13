from src.lsh.lsh import LSH
import src.cnn.contrastiveLoss as cl
import pickle
import src.lsh.lsh_testing as lt

def main():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    print(model)

    #NOTE: Only run this if you want to retrain the model. Else, the loaded model is already trained
    #cl.train_model(model, cl.train_loader, cl.val_loader, cl.criterion, cl.optimizer, epochs=8)
    
    accuracy, avgTime1 = cl.evaluate(model, cl.testReal_loader, cl.real_loader, cl.fake_loader, threshold=0.5)
    print()
    accuracy2, avgTime2 = cl.evaluate(model, cl.testFake_loader, cl.real_loader, cl.fake_loader, threshold=0.5)
    accuracy2 = 1- accuracy2
    print(f"Accuracy of labeling real test images as real: {accuracy}")
    print(f"Accuracy of labeling fake test images as fake: {accuracy2}")
    print()
    avgTime = (avgTime1 + avgTime2)/2
    print(f"Average time taken per prediction (in milliseconds): {avgTime}")
    print()

    cl.save_embeddings(model, cl.real_loader, "embeddings/real_embeddings.txt", cl.device)
    cl.save_embeddings(model, cl.fake_loader, "embeddings/fake_embeddings.txt", cl.device)
    cl.save_embeddings(model, cl.testReal_loader, "embeddings/test_real_embeddings.txt", cl.device)
    cl.save_embeddings(model, cl.testFake_loader, "embeddings/test_fake_embeddings.txt", cl.device)

    real_embeddings_df = lt.embeddings_to_dataframe("src/embeddings/real_embeddings.txt")
    fake_embeddings_df = lt.embeddings_to_dataframe("src/embeddings/fake_embeddings.txt")
    test_real_embeddings_df = lt.embeddings_to_dataframe("src/embeddings/test_real_embeddings.txt")
    test_fake_embeddings_df = lt.embeddings_to_dataframe("src/embeddings/test_fake_embeddings.txt")

    dimension = real_embeddings_df.shape[1] 


    lsh = LSH(num_hash_funcs=10, threshold=0.99, dimension=dimension)


    accuracy_real, accuracy_fake, avg_time = lt.evaluate_lsh(lsh, test_real_embeddings_df, test_fake_embeddings_df)

    print(f"Accuracy of labeling real test embeddings as real: {accuracy_real}")
    print(f"Accuracy of labeling fake test embeddings as fake: {accuracy_fake}")
    print()
    print(f"Average time per prediction (in milliseconds): {avg_time}")

if __name__ == "__main__":
    main()
