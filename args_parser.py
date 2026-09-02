import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Args for ex3: Fourier Transform and Individual Alpha Frequency',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    
    # General parameters
    parser.add_argument('--elec_num', 
                        type=int, 
                        default=18, 
                        help='Pz electrode number (0-indexed)')
    
    parser.add_argument('--DATA_DIR', 
                        type=str, 
                        default='data', 
                        help='Data directory path')
    
    parser.add_argument('--window_size', 
                        type=int, 
                        default=4,
                        help='Window size [seconds] for Welch methods')
    
    parser.add_argument('--overlap', 
                        type=float, 
                        default=2,
                        help='Overlap [seconds] for Welch methods')
    
    return parser.parse_args()