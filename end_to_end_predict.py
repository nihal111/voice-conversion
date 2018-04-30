import argparse
from predict import predict_phonemes, load_vocab
from predict2 import predict_mags, converter


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='input file path')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    predict_file = args.input_file

    outputs = predict_phonemes(predict_file)
    print(outputs)
    outputs = predict_mags(outputs)
    converter(outputs, "end2end.wav")
