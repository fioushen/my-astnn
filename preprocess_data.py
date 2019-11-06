from pycparser import c_parser
import pandas as pd
from util import *
import util_java
import util_c
import javalang
from gensim.models.word2vec import Word2Vec


class Preprocessor:
    r"""Data Preprocessor for astnn.

    Given a raw data, it outputs precessed data.

    Input raw data format: a DataFrame object with 3 column: id, code, label
    Output data format: id, code, label, ast, index_tree, token_seq
    """

    def __init__(self, raw_data_path, language, w2v_path, output_path, entry_num=-1):
        self.raw_data_path = raw_data_path
        if language == 'c':
            self.parseFunc = c_parser.CParser().parse
            self.get_ast_nodes_func = util_c.get_ast_nodes
            self.ast2sequence_func = util_c.ast2sequence
        elif language == 'java':
            def parse_program(func):
                tokens = javalang.tokenizer.tokenize(func)
                parser = javalang.parser.Parser(tokens)
                tree = parser.parse_member_declaration()
                return tree
            self.parseFunc = parse_program
            self.get_ast_nodes_func = util_java.get_ast_nodes
            self.ast2sequence_func = util_java.ast2sequence
        else:
            raise ValueError('Parameter language must be "c" or "java".')
        self.w2v_path = w2v_path
        self.output_path = output_path
        self.entry_num = entry_num

    def run(self):
        # read data, it's a DataFrame with 3 columns:
        # id(int), code(code text), label(int)
        print('Reading data...')
        programs = pd.read_pickle(self.raw_data_path)[:self.entry_num]
        programs.columns = ['id', 'code', 'label']

        print('Parsing to AST...')
        programs['ast'] = programs['code'].apply(self.parseFunc)

        # transform ast to a sequence of symbols of AST
        corpus = programs['ast'].apply(self.ast2sequence_func)

        print('Training word embedding...')
        w2v = Word2Vec(corpus, size=128, workers=16, sg=1, min_count=3)  # use w2v[WORD] to get embedding
        vocab = w2v.wv.vocab

        # transform ASTNode to tree of index in word2vec model
        def node_to_index(node):
            result = [vocab[node.token].index if node.token in vocab else len(vocab)]
            for child in node.children:
                result.append(node_to_index(child))
            return result

        # transform ast to trees of index in word2vec model
        def ast_to_index(ast):
            blocks = []
            self.get_ast_nodes_func(ast, blocks)
            return [node_to_index(b) for b in blocks]

        def index_to_tokens(index_tree):
            return [w2v.wv.index2word[index] if index < len(vocab) else 'UNKNOWN' for index in flatten_tree(index_tree)]

        print('Transforming ast to embedding index tree...')
        programs['index_tree'] = programs['ast'].apply(ast_to_index)

        programs['token_seq'] = programs['index_tree'].apply(index_to_tokens)

        w2v.save(self.w2v_path)
        print("Saved word2vec model at", self.w2v_path)
        programs.to_pickle(self.output_path)
        print("Saved processed data at", self.output_path)


if __name__ == '__main__':
    preprocessor = Preprocessor(raw_data_path='./data/c/id-code-label.pkl', language='c',
                                w2v_path='./data/c/w2v_128', output_path='./data/c/id_code_label_ast_(index_tree).pkl',
                                entry_num=-1)
    preprocessor.run()
