import random
from typing import Dict
from .utility import _hash
from .config import DELIMITERS
import re

def build_plain_prompt(messages: list[Dict], delimiter: str) -> str:
    instruction = ""
    input_parts = []

    for m in messages:
        role    = m["role"]
        content = m["content"]
        if content is None:
            continue

        if role == "system":
            instruction += content + "\n"
        else:  ## fixme: I add user task as instruction as well
            input_parts.append(content)

    instruction = instruction.strip()
    input_text = "\n".join(input_parts).strip()

    # standard delimiting
    plain_prompt = (
        DELIMITERS[delimiter][0]
        + f"\n{instruction}\n\n"
        + DELIMITERS[delimiter][1]
        + f"\n{input_text}\n\n"
        + DELIMITERS[delimiter][2]
        + "\n"
    )
    return plain_prompt

class Prompt:
    def __init__(self, instruction: str, delimiter: str, system=None):
        self.instruction = instruction
        self.system = system
        self.delimiter = delimiter
        
    def __call__(self, tokenizer, return_dict=False):
        din = []
        
        if self.system:
            din.append({"role":"system", "content":self.system})
        din.append({"role":'user', "content": self.instruction})
                    
        return build_plain_prompt(din, self.delimiter)
    

class AdvPrompt:
    
    def __init__(self, payload, target, honest_input, delimiter):
        self.payload = payload.strip()
        self.target = {k:v.strip() for (k, v) in target.items()}
        self.honest_input = honest_input
        
        # default
        self.type_adv_seg = 0
        self.adv_content = None
    
        self.split_char = '.'

        self.delimiter = delimiter

    def make_adv_content(self, ne, adv_placeholder_token):
        prefix  = adv_placeholder_token * ne.prefix_size
        postfix = adv_placeholder_token * ne.postfix_size
        return ne.sep + prefix + self.payload + postfix + ne.sep
    
            
    def inject_in_honest_text(self, adv_content, honest_text, adv_pos, replace=True):
        
        sentences = honest_text.split(self.split_char)
        sentences = [s+self.split_char for s in sentences if s]
        
        k = len(sentences) - replace    
        
        # if there is a single piece of text, either append or prepend
        if k == 0:
            k = 1 
            replace = False
            
        if adv_pos is None:
            # random but per run deterministic
            adv_pos = _hash(self.payload) % k
        elif adv_pos < 0:
            # random
            adv_pos = random.randint(0, k)
            
        _adv_content = adv_content
        
        if replace:
            sentences[adv_pos] = _adv_content
        else:
            sentences.insert(adv_pos, _adv_content)
            
        text = ''.join(sentences)

        return text, adv_pos

########################################################################################      
        
class SingleInputPrompt(AdvPrompt):
    def __init__(
        self,
        payload,
        target,
        honest_input,
        template,
        system_prompt,
        delimiter
    ):
        AdvPrompt.__init__(self, payload, target, honest_input, delimiter)
    
        self.template = template
        self.system_prompt = system_prompt
        self.delimiter = delimiter

    
    def __call__(self, llm, ne, with_target=True, adv_pos=None, return_dict=False):

        if self.adv_content is None:
            adv_content = self.make_adv_content(ne, llm.adv_placeholder_token)
        else:
            adv_content = self.adv_content
            
        text, _ = self.inject_in_honest_text(adv_content, self.honest_input, adv_pos, replace=False)
       
        task = self.template.format(text=text)
        query = re.split(r'\{[^}]+\}', self.template, 1)[0]
        instruction = self.system_prompt+task.split(query)[0] + query
        data = task.split(query)[1]

        if return_dict:
            d = []
            if self.system_prompt:
                d.append({"role":'system', 'content': instruction})
            d.append({"role":'user', 'content': data})
            
            return d
        else:
        
            prompt = Prompt(data, self.delimiter, instruction)(llm.tokenizer)

            if with_target:
                prompt += self.target[llm.llm_name]

        
        return prompt

########################################################################################      

class MultiInputPrompt(AdvPrompt):
    def __init__(
        self,
        payload,
        target,
        honest_input,
        template,
        system_prompt,
        delimiter,
    ):
        AdvPrompt.__init__(self, payload, target, honest_input, delimiter)
        self.adv_content = None
    
        self.template, self.input_template, self.suffix = template
        self.system_prompt = system_prompt
        self.delimiter = delimiter

    def make_adv_chunk(self, adv_seg, adv_pos):
        
        _adv_pos = adv_pos
        
        if adv_pos is None:
            # random but per run deterministic
            adv_pos = _hash(self.payload) % (len(self.honest_input)-1)
        elif adv_pos < 0:
            # random
            adv_pos = random.randint(0, len(self.honest_input)-1)
                    
        chunk_to_inject = self.honest_input[adv_pos]
        
        adv_chunk, vector_pos = self.inject_in_honest_text(adv_seg, chunk_to_inject, _adv_pos)
        
        if vector_pos != 0:
            # if adv_chunk does not start with armed payload, add space 
            adv_chunk = adv_chunk
        
        return adv_chunk
    
    def __call__(self, llm, ne, with_target=True, adv_pos=None, return_dict=False):
        
        if self.adv_content is None:
            adv_content = self.make_adv_content(ne, llm.adv_placeholder_token)
        else:
            adv_content = self.adv_content
        adv_content = self.make_adv_chunk(adv_content, adv_pos)
        
        number_of_honest_inputs = _hash(''.join(self.honest_input) + self.payload) % len(self.honest_input) + 1
        body = [self.input_template.format(text=' '+text) for text in self.honest_input[:number_of_honest_inputs]]
        adv_input = self.input_template.format(text=adv_content) 
        
        if adv_pos is None:
            # random but per run deterministic
            adv_pos = _hash(self.payload) % len(self.honest_input)
        elif adv_pos < 0:
            # random
            adv_pos = random.randint(0, len(self.honest_input))
        
        body.insert(adv_pos, adv_input)
        
        text = ''.join(body)

        task = self.template.format(text=text)
        task += self.suffix

        query = re.split(r'\{[^}]+\}', self.template, 1)[0]
        instruction = self.system_prompt+task.split(query)[0] + query
        data = task.split(query)[1]
        
        if return_dict:
            d = []
            if self.system_prompt:
                d.append({"role":'system', 'content': instruction})
            d.append({"role":'user', 'content': data})
            
            return d
        else:
        
            prompt = Prompt(data, self.delimiter, instruction)(llm.tokenizer)

            if with_target:
                prompt += self.target[llm.llm_name]


        return prompt
        
########################################################################################      
      
class QAPrompt(MultiInputPrompt):
    @staticmethod
    def make_source(c):
        return str(random.randint(0, 10000000))
    
    def __init__(
        self,
        payload,
        target,
        honest_input,
        query,
        system_prompt,
        delimiter
    ):
        AdvPrompt.__init__(self, payload, target, honest_input, delimiter)
        self.adv_content = None
    
        self.query = query
        
        self.template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").\nIf you don't know the answer, just say that you don't know. Don't try to make up an answer. ALWAYS return a "SOURCES" part in your answer.\n\nQUESTION: {query}\n========={text}\n========="""
        self.input_template = """\nContent:{text}\nSource: {source}"""
        self.suffix = '\nFINAL ANSWER: '
        self.system_prompt = system_prompt
        self.delimiter = delimiter

        
    def __call__(self, llm, ne, with_target=True, adv_pos=None, return_dict=False):

        if self.adv_content is None:
            adv_content = self.make_adv_content(ne, llm.adv_placeholder_token)
        else:
            adv_content = self.adv_content
        adv_content = self.make_adv_chunk(adv_content, adv_pos)

        number_of_honest_inputs = _hash(''.join(self.honest_input) + self.payload) % len(self.honest_input) + 1
        body = [self.input_template.format(text=' '+text, source=self.make_source(text)) for text in self.honest_input[:number_of_honest_inputs]]
        adv_input = self.input_template.format(text=adv_content, source=self.make_source(self.payload)) 
        
        if adv_pos is None:
            # random but per run deterministic
            adv_pos = _hash(self.payload) % len(self.honest_input)
        elif adv_pos < 0:
            # random
            adv_pos = random.randint(0, len(self.honest_input))
        
        body.insert(adv_pos, adv_input)
        
        text = ''.join(body)
        task = self.template.format(query=self.query, text=text)
        task += self.suffix

        instruction = self.system_prompt+task.split(self.query)[0] + self.query
        data = task.split(self.query)[1]
               
        if return_dict:
            d = []
            if self.system_prompt:
                d.append({"role":'system', 'content':instruction})
            d.append({"role":'user', 'content':data})
            
            return d
        else:
        
            prompt = Prompt(data, self.delimiter, instruction)(llm.tokenizer)

            if with_target:
                prompt += self.target[llm.llm_name]

        
        return prompt
  

########################################################################################      

class CodePrompt(SingleInputPrompt):
    def __init__(self, *args, **kargs):
        SingleInputPrompt.__init__(self, *args, **kargs)
        self.split_char = '\n'
    
        
    def __call__(self, llm, ne, with_target=True, adv_pos=None, return_dict=False):
                
        if self.adv_content is None:
            adv_content = self.make_adv_content(ne, llm.adv_placeholder_token)
        else:
            adv_content = self.adv_content
            
        text, _ = self.inject_in_honest_text(adv_content, self.honest_input, adv_pos, replace=False)
       
        task = self.template.format(text=text)
        query = re.split(r'\{[^}]+\}', self.template, 1)[0]
        instruction = self.system_prompt+task.split(query)[0] + query
        data = task.split(query)[1]
                
        if return_dict:
            d = []
            if self.system_prompt:
                d.append({"role":'system', 'content': instruction})
            d.append({"role":'user', 'content': data})
        
            return d
        else:
        
            prompt = Prompt(data, self.delimiter, instruction)(llm.tokenizer)

            if with_target:
                prompt += self.target[llm.llm_name]

        
        return prompt