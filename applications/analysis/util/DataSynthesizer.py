import sdv.metadata
from sdv.single_table.ctgan import CTGANSynthesizer

from sdv.metadata import Metadata, SingleTableMetadata
from sdv.datasets.local import load_csvs
from sdv.evaluation.single_table import evaluate_quality


def driver():
    metadata_source: sdv.metadata.SingleTableMetadata = SingleTableMetadata()
    metadata_source.detect_from_csv("/Users/krishnaramachandran/Downloads/merged_balance_income/merged.csv")
    print(metadata_source)
    datasets = load_csvs("/Users/krishnaramachandran/Downloads/merged_balance_income/",
                         read_csv_parameters={'skipinitialspace': True})
    data = datasets['merged']
    synthesizer = CTGANSynthesizer(
        metadata_source,
        enforce_rounding=False,
        epochs=1900,
        verbose=True
    )

    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(100000)
    synthetic_data.to_csv("/Users/krishnaramachandran/Downloads/synthetic", index=None, header=True)
    quality_report = evaluate_quality(real_data=data, synthetic_data=synthetic_data, metadata=metadata_source)
    quality_report = evaluate_quality(real_data=data, synthetic_data=synthetic_data, metadata=metadata_source)


if __name__ == "__main__":
    driver()

