import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TrainDataset(Dataset):
    """
    Dataset for LLM fine-tuning on pairwise response comparison.

    Args:
        dataset (DataFrame): The dataset containing rows with 'prompt', 'response_a', 'response_b', and 'winner'.
        tokenizer: The tokenizer to process input strings.
        cfg: Configuration object or dictionary containing max lengths.
        preprocess_fn (callable): A preprocessing function to process each row.
    """

    def __init__(self, dataset, tokenizer, cfg, preprocess_fn=None):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.dataset = self.process2(self.dataset, tokenizer)
        self.tokenizer = tokenizer
        
        self.preprocess_fn = preprocess_fn or self.default_preprocess_fn  # Use custom preprocess if provided

    def __len__(self):
        return len(self.dataset)

    def default_preprocess_fn(self, row):
        """
        Default preprocessing function to handle a single row.

        Args:
            row (dict): A single row of the dataset.

        Returns:
            input_ids (list[int]): Tokenized input IDs.
            attention_mask (list[int]): Attention mask for the input.
            length (int): The length of the input sequence.
        """
        prompt = '<prompt>: ' + row['prompt']

        response_a = '\n\n<response_a>: ' + row['response_a']
        response_b = '\n\n<response_b>: ' + row['response_b']

        # Tokenize prompt and responses
        p = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
        a = self.tokenizer(response_a, add_special_tokens=False)['input_ids']
        b = self.tokenizer(response_b, add_special_tokens=False)['input_ids']

        # # Truncate prompt if it exceeds max length
        if len(p) > self.cfg.max_prompt_length:
            p = p[-self.cfg.max_prompt_length:]

        # # Calculate response lengths
        response_length = (self.cfg.max_length - len(p)) // 2
        # response_length = self.cfg.max_length
        # Build input_ids with special tokens
        input_ids = (
            [self.tokenizer.bos_token_id] +
            p +
            a[-response_length:] +
            b[-response_length:] +
            [self.tokenizer.eos_token_id]
        )

        # Build attention mask
        length = len(input_ids)
        attention_mask = [1] * length

        return input_ids, attention_mask, length

    def process2(self, row, tokenizer):
        for col in ['prompt', 'response_a', 'response_b']:
            row[col] = row[col].fillna('')
            text_list = []
            if col == 'prompt':
                max_no = 512
                s_no = 256
                e_no = -256
            else:
                max_no = self.cfg.max_length
                s_no = max_no // 2
                e_no = -max_no // 2
            for text in tqdm(row[col]):
                encoded = tokenizer(text, return_offsets_mapping=True)
                if len(encoded['input_ids']) > max_no:
                    start_idx, end_idx = encoded['offset_mapping'][s_no]
                    new_text = text[:end_idx]
                    # print(len(tokenizer(text[:end_idx])['input_ids']))
                    start_idx, end_idx = encoded['offset_mapping'][e_no]
                    # print(len(tokenizer(text[start_idx:])['input_ids']))
                    new_text = new_text + "\n(snip)\n" + text[start_idx:]
                    # print(len(tokenizer(new_text)['input_ids']), new_text)
                    text = new_text
                text_list.append(text)
            row[col] = text_list
        return row

                      

    def __getitem__(self, idx):
        """
        Get a single processed example.

        Args:
            idx (int): Index of the example.

        Returns:
            dict: A dictionary containing processed input IDs, attention mask, and label.
        """
        row = self.dataset.iloc[idx]
        input_ids, attention_mask, length = self.preprocess_fn(row)

        # Convert winner ('model_a' or 'model_b') to binary label
        label = 0 if row['winner'] == 'model_a' else 1

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long)
        }


def collate_fn(batch):
    """
    Collate function to create mini-batches.

    Args:
        batch (list[dict]): A list of examples.

    Returns:
        dict: A batch containing padded input IDs, attention masks, and labels.
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    lengths = [item['length'] for item in batch]

    # Padding
    max_length = max(lengths)
    padded_input_ids = torch.stack([
        torch.cat([ids, torch.zeros(max_length - len(ids), dtype=torch.long)]) for ids in input_ids
    ])
    padded_attention_mask = torch.stack([
        torch.cat([mask, torch.zeros(max_length - len(mask), dtype=torch.long)]) for mask in attention_mask
    ])
    labels = torch.stack(labels)

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'labels': labels,
    }

if __name__ == "__main__":
    from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast
    import pandas as pd
    from config import get_config 
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    config = get_config()
    train_data = pd.read_csv('/root/autodl-tmp/WSDM/input/train.csv')[20000:]
    train_data['response_b'] = train_data.apply(
        lambda row: '\n\n<response_b>: ' + (str(row['response_b']) if pd.notnull(row['response_b']) else 'N/A'),
        axis=1
    )
    train_data['response_a'] = train_data.apply(
        lambda row: '\n\n<response_a>: ' + (str(row['response_a']) if pd.notnull(row['response_a']) else 'N/A'),
        axis=1
    )
    tokenizer = GemmaTokenizerFast.from_pretrained(config.model_name_or_path)
    train_dataset = TrainDataset(train_data, tokenizer, config)
    train_dataloader = DataLoader(train_dataset, 
                                shuffle=False,
                                batch_size=config.per_device_train_batch_size, 
                                collate_fn=collate_fn,
                                pin_memory=True
                                )
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch")):
        pass


