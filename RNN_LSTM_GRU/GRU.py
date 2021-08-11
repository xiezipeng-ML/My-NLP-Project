import oneflow as flow
from oneflow import nn
from oneflow import Tensor
from oneflow.nn import Parameter
from math import sqrt


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        gate_size = 3 * hidden_size
        self.drop = nn.Dropout(self.dropout)

        for layer in range(num_layers):
            for direction in range(num_directions):
                
                real_hidden_size = hidden_size
                layer_input_size = input_size if layer == 0 else real_hidden_size * num_directions
                
                w_ih = Parameter(Tensor(gate_size, layer_input_size))
                w_hh = Parameter(Tensor(gate_size, real_hidden_size))
                b_ih = Parameter(Tensor(gate_size))
                b_hh = Parameter(Tensor(gate_size))

                layer_params = ()
                
                if bias:
                    layer_params = (w_ih, w_hh, b_ih, b_hh)
                else:
                    layer_params = (w_ih, w_hh)

                suffix = '_reverse' if direction ==1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.uniform_(-stdv, stdv)
    
    def permute_tensor(self, input):
        return input.permute(1,0,2)

    def forward(self, x, hidden=None):
        if self.batch_first == False:
            x = self.permute_tensor(x)
        D = 2 if self.bidirectional else 1
        num_layers = self.num_layers
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            h_t = flow.zeros((D*num_layers,batch_size, self.hidden_size), dtype=x.dtype, device=x.device)
        else:
            h_t = hidden

        if self.bidirectional:
            h_t_f = flow.cat([h_t[l, :, :].unsqueeze(0) for l in range(h_t.size(0)) if l%2==0],dim=0)
            h_t_b = flow.cat([h_t[l, :, :].unsqueeze(0) for l in range(h_t.size(0)) if l%2!=0],dim=0)
        else:
            h_t_f = h_t
        
        layer_hidden = []

        for layer in range(self.num_layers):
            hidden_seq_f = []
            if self.bidirectional:
                hidden_seq_b = []

            hid_t_f = h_t_f[layer, :, :]
            if self.bidirectional:
                hid_t_b = h_t_b[layer, :, :]
            
            for t in range(seq_len):
                if layer == 0:
                    x_t_f = x[:, t, :]
                    if self.bidirectional:
                        x_t_b = x[:, seq_len-1-t, :]
                else:
                    x_t_f = hidden_seq[:, t, :] 
                    if self.bidirectional:
                        x_t_b = hidden_seq[:, seq_len-1-t, :]

    # 前向
                gi_f = flow.matmul(x_t_f, getattr(self, 'weight_ih_l{}{}'.format(layer,'')).permute(1,0))
                gh_f = flow.matmul(hid_t_f, getattr(self, 'weight_hh_l{}{}'.format(layer,'')).permute(1,0))
                if self.bias:
                    gi_f += getattr(self, 'bias_ih_l{}{}'.format(layer,''))
                    gh_f += getattr(self, 'bias_hh_l{}{}'.format(layer,''))
                
                i_r_f, i_i_f, i_n_f = gi_f.chunk(3, dim=1)
                h_r_f, h_i_f, h_n_f = gh_f.chunk(3, dim=1)

                resetgate_f = flow.sigmoid(i_r_f + h_r_f)
                inputgate_f = flow.sigmoid(i_i_f + h_i_f)
                newgate_f = flow.tanh(i_n_f + resetgate_f * h_n_f)

                hid_t_f = newgate_f + inputgate_f * (hid_t_f - newgate_f)

                hidden_seq_f.append(hid_t_f.unsqueeze(1))         
    
    # 后向      
                if self.bidirectional:
                    gi_b = flow.matmul(x_t_b, getattr(self, 'weight_ih_l{}{}'.format(layer,'_reverse')).permute(1,0))
                    gh_b = flow.matmul(hid_t_b, getattr(self, 'weight_hh_l{}{}'.format(layer,'_reverse')).permute(1,0))
                    if self.bias:
                        gi_b += getattr(self, 'bias_ih_l{}{}'.format(layer,'_reverse'))
                        gh_b += getattr(self, 'bias_hh_l{}{}'.format(layer,'_reverse'))
                    
                    i_r_b, i_i_b, i_n_b = gi_b.chunk(3, dim=1)
                    h_r_b, h_i_b, h_n_b = gh_b.chunk(3, dim=1)

                    resetgate_b = flow.sigmoid(i_r_b + h_r_b)
                    inputgate_b = flow.sigmoid(i_i_b + h_i_b)
                    newgate_b = flow.tanh(i_n_b + resetgate_b * h_n_b)

                    hid_t_b = newgate_b + inputgate_b * (hid_t_b - newgate_b)

                    hidden_seq_b.insert(0, hid_t_b.unsqueeze(1))     
            
            
            hidden_seq_f = flow.cat(hidden_seq_f, dim=1)    
            if self.bidirectional:
                hidden_seq_b = flow.cat(hidden_seq_b, dim=1)    
            
            if self.dropout != 0 and layer != self.num_layers-1:
                hidden_seq_f = self.drop(hidden_seq_f)
                if self.bidirectional:
                    hidden_seq_b = self.drop(hidden_seq_b)

            if self.bidirectional:
                hidden_seq = flow.cat([hidden_seq_f, hidden_seq_b], dim=2)   
            else:
                hidden_seq = hidden_seq_f

            if self.bidirectional:
                h_t = flow.cat([hid_t_f.unsqueeze(0), hid_t_b.unsqueeze(0)], dim=0)   
            else:
                h_t = hid_t_f.unsqueeze(0)

            layer_hidden.append(h_t)        
            
        h_t = flow.cat(layer_hidden, dim=0)    
        
        if self.batch_first == False:
            hidden_seq = self.permute_tensor(hidden_seq)

        return hidden_seq, h_t