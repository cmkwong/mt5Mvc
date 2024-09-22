import torch

def getCosineSimilarity(tt, dim=1):
    """
    tt: torch tensor
    """
    magnitudes = tt.pow(2).sum(dim=dim).sqrt().unsqueeze(0)
    similarities = torch.mm(tt, tt.t()) / magnitudes
    return similarities

# applied in embedding vector
def getNegativeSamplingLoss(input_vectors, output_vectors, noise_vectors):
    batch_size, embed_size = input_vectors.shape

    # Input vectors should be a batch of column vectors
    input_vectors = input_vectors.view(batch_size, embed_size, 1)

    # Output vectors should be a batch of row vectors
    output_vectors = output_vectors.view(batch_size, 1, embed_size)

    # bmm = batch matrix multiplication
    # correct log-sigmoid loss
    out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
    out_loss = out_loss.squeeze()

    # incorrect log-sigmoid loss
    noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
    noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors

    # negate and sum correct and noisy log-sigmoid losses
    # return average batch loss
    return -(out_loss + noise_loss).mean()