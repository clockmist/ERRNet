from options.errnet.train_options import TrainOptions


class TrainOptionsV2(TrainOptions):
    """Extended options for ERRNet V2 with Transformer + Frequency Enhancement."""

    def initialize(self):
        TrainOptions.initialize(self)

        # Transformer backbone settings
        self.parser.add_argument(
            '--block_type', type=str, default='transformer',
            choices=['residual', 'transformer'],
            help='type of backbone blocks (residual=original, transformer=Restormer-style)')
        self.parser.add_argument(
            '--n_transformer_blocks', type=int, default=8,
            help='number of TransformerBlocks in the bottleneck')
        self.parser.add_argument(
            '--n_heads', type=int, default=4,
            help='number of attention heads in MDTA')

        # Frequency enhancement module
        self.parser.add_argument(
            '--use_frequency_module', action='store_true', default=True,
            help='enable FrequencyEnhancementModule')
        self.parser.add_argument(
            '--no_frequency_module', action='store_false', dest='use_frequency_module',
            help='disable FrequencyEnhancementModule')
        self.parser.add_argument(
            '--lambda_fft', type=float, default=0.1,
            help='weight for FFT loss (0 to disable)')
