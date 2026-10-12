# MLOps Micro-Credential: Course Materials (October 2026)

Code and materials for the Translational AI Center's MLOps micro-credential. The course runs on the Jetstream2 cloud (and later the NCSA Delta cluster). **This repository currently contains Module 1 only**; later modules will be added as they are released.

## What's released

| Module | Folder | Platform | Status |
|--------|--------|----------|--------|
| 1. Data pipelines | [`01_Jetstream2_DataPipelines`](01_Jetstream2_DataPipelines/) | Jetstream2 | Released |

## Module 1: Data pipelines

You will containerize a Streamlit app that extracts frames from uploaded video, run Label Studio to annotate those frames, and then orchestrate both with an Airflow DAG.

**Start here:** [`01_Jetstream2_DataPipelines/README.md`](01_Jetstream2_DataPipelines/README.md) has the full setup, commands, and a table of the ports you use to reach each app from your own computer.

## Getting started on Jetstream2

1. Log in at the [Jetstream2 Exosphere portal](https://docs.jetstream-cloud.org/ui/exo/login/).
2. Launch an instance (your instructor will tell you the image and flavor).
3. Open the web shell or web desktop, or connect over SSH as `exouser`.
4. Clone this repository into your home directory:
   ```bash
   cd ~
   git clone https://github.com/TranslationalAICenterISU/mlops-oct-2026.git
   cd mlops-oct-2026
   ```
5. Follow the Module 1 README.

> **Heads up on folder names:** the Module 1 commands and the Airflow DAG assume the folder lives at `/home/exouser/01_Jetstream2_DataPipelines`. If you clone the repo as above, copy or symlink the folder there:
> `ln -s ~/mlops-oct-2026/01_Jetstream2_DataPipelines ~/01_Jetstream2_DataPipelines`

## Prerequisites

- Jetstream2 access (provided through Canvas / your ACCESS allocation)
- Basic Python and machine learning familiarity
- Docker and Airflow are covered in the live sessions

## License

MIT. See [LICENSE](LICENSE).

## Contact

Questions? Contact the course instructor or open an issue in this repository.

## Acknowledgments

- Translational AI Center at Iowa State University
- Jetstream2 cloud computing resource
