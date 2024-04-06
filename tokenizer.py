import torch

def token_auto_concat_embeds(pipe, positive, negative):
    max_length = pipe.tokenizer.model_max_length
    positive_length = pipe.tokenizer(positive, return_tensors="pt").input_ids.shape[-1]
    negative_length = pipe.tokenizer(negative, return_tensors="pt").input_ids.shape[-1]
    
    print(f'Token length is model maximum: {max_length}, positive length: {positive_length}, negative length: {negative_length}.')
    if max_length < positive_length or max_length < negative_length:
        print('Concatenated embedding.')
        if positive_length > negative_length:
            positive_ids = pipe.tokenizer(positive, return_tensors="pt").input_ids.to("cuda")
            negative_ids = pipe.tokenizer(negative, truncation=False, padding="max_length", max_length=positive_ids.shape[-1], return_tensors="pt").input_ids.to("cuda")
        else:
            negative_ids = pipe.tokenizer(negative, return_tensors="pt").input_ids.to("cuda")  
            positive_ids = pipe.tokenizer(positive, truncation=False, padding="max_length", max_length=negative_ids.shape[-1],  return_tensors="pt").input_ids.to("cuda")
    else:
        positive_ids = pipe.tokenizer(positive, truncation=False, padding="max_length", max_length=max_length,  return_tensors="pt").input_ids.to("cuda")
        negative_ids = pipe.tokenizer(negative, truncation=False, padding="max_length", max_length=max_length, return_tensors="pt").input_ids.to("cuda")
    
    positive_concat_embeds = []
    negative_concat_embeds = []
    for i in range(0, positive_ids.shape[-1], max_length):
        positive_concat_embeds.append(pipe.text_encoder(positive_ids[:, i: i + max_length])[0])
        negative_concat_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])
    
    positive_prompt_embeds = torch.cat(positive_concat_embeds, dim=1)
    negative_prompt_embeds = torch.cat(negative_concat_embeds, dim=1)
    return positive_prompt_embeds, negative_prompt_embeds