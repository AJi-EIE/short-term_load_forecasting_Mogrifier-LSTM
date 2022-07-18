from mog_lstm import MogrifierLSTMCell
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
class MogLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, mogrify_steps, output_size, batch_size):
        super(MogLSTM, self).__init__()
        self.hidden_size = hidden_size  # 7
        self.batch_size = batch_size   # lstm_batch_size  30
        self.input_size = input_size   # 7
        self.output_size = output_size  # 1
        self.mogrifier_lstm_layer1 = MogrifierLSTMCell(input_size, hidden_size, mogrify_steps)
        self.mogrifier_lstm_layer2 = MogrifierLSTMCell(hidden_size, hidden_size, mogrify_steps)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_seq):

        batch_size = input_seq.shape[1]   #24

        input_seq = input_seq.view(self.batch_size, batch_size, self.input_size)   #len 30， batch_size 24 , inputsize 7

        h1,c1 = [torch.zeros(self.batch_size,self.hidden_size).to(device), torch.zeros(self.batch_size,self.hidden_size).to(device)]
        h2,c2 = [torch.zeros(self.batch_size,self.hidden_size).to(device), torch.zeros(self.batch_size,self.hidden_size).to(device)]

        hidden_states = []
        outputs = []
        
        for step in range(batch_size):

            x = input_seq[:, step]
            h1,c1 = self.mogrifier_lstm_layer1(x, (h1, c1))     
            h2,c2 = self.mogrifier_lstm_layer2(h1, (h2, c2))
            out = self.linear(h2)

            hidden_states.append(h2.unsqueeze(1))
            outputs.append(out.unsqueeze(1))    #24
        

        hidden_states = torch.cat(hidden_states, dim = 1)  
        outputs = torch.cat(outputs, dim = 1)    # 30,24,1
        outputs = outputs[:, -1, :]
       
        
        return outputs


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):

        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)

        seq_len = input_seq.shape[1]

        input_seq = input_seq.view(self.batch_size, seq_len, self.input_size)    # (30 * 24, 7)

        output, _ = self.lstm(input_seq, (h_0, c_0))
        output = output.contiguous().view(self.batch_size * seq_len, self.hidden_size)  # (30 * 24, 7)

        pred = self.linear(output)  
        pred = pred.view(self.batch_size, seq_len, -1)
        pred = pred[:, -1, :]


        return pred


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):

        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)

        seq_len = input_seq.shape[1]
        input_seq = input_seq.view(self.batch_size, seq_len, self.input_size)

        output, _ = self.lstm(input_seq, (h_0, c_0))
        output = output.contiguous().view(self.batch_size, seq_len, self.num_directions, self.hidden_size)
        output = torch.mean(output, dim=2)
        output = output.contiguous().view(self.batch_size * seq_len, self.hidden_size)  # (5 * 30, 64)

        pred = self.linear(output)  # pred()
        pred = pred.view(self.batch_size, seq_len, -1)
        pred = pred[:, -1, :]


        return pred



class CNNLSTMModel(nn.Module):

    # def __init__(self, window=5, dim=4, lstm_units=16, num_layers=2):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, window=1):
        super(CNNLSTMModel, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.conv1d = nn.Conv1d(input_size, self.hidden_size, 1)     # 7,30 ,24
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, num_layers=num_layers, bidirectional=False)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(self.hidden_size, output_size)
        self.act4 = nn.Tanh()

    def forward(self, x):

        seq_len = x.shape[1]
        x = x.view(self.batch_size, self.input_size, seq_len)    #30,7,24

        x = self.conv1d(x)    # 30, 30, 22
        x = self.act1(x)           
        x = self.maxPool(x) 
        x = self.drop(x)     # 30, 30, 22
        x = x.view(self.batch_size, seq_len, self.hidden_size) 
        x, (_, _) = self.lstm(x)     #30 24 32
        output = x.contiguous().view(self.batch_size * seq_len, self.hidden_size)

        
        pred = self.cls(output)  
        pred = pred.view(self.batch_size, seq_len, -1)
        pred = pred[:, -1, :]

        return pred