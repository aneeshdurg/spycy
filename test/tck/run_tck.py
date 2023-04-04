import glob
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from multiprocessing.pool import ThreadPool


def run_feature_class(args):
    feature_class, junit_dir = args
    print("Running", feature_class, file=sys.stderr)
    subprocess.run(
        [
            "behave",
            "--quiet",
            "-i",
            f"{feature_class}[0-9]",
            "--junit",
            "--junit-directory",
            junit_dir,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("Completed", feature_class, file=sys.stderr)


def main(update: bool):
    failures = set()

    gitroot = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode()
    os.environ["PYTHONPATH"] = gitroot.strip()
    features = glob.glob("features/**/*.feature", recursive=True)
    extract_name = re.compile(".*/([a-zA-Z]*)\d+")
    feature_classes = set(extract_name.match(feature).group(1) for feature in features)
    with tempfile.TemporaryDirectory() as tmpdir:
        pool = ThreadPool(processes=16)
        pool.map(
            run_feature_class,
            [(feature_class, tmpdir) for feature_class in feature_classes],
        )
        pool.close()
        for f in os.listdir(tmpdir):
            tree = ET.parse(f"{tmpdir}/{f}")
            for testcase in tree.findall("testcase"):
                if testcase.get("status") == "failed":
                    name = testcase.get("name")
                    assert name
                    test_name = f"{f}:{name}"
                    failures.add(test_name.strip())

    exit_status = 0
    if update:
        with open("expected_failures.txt", "w") as f:
            f.write("\n".join(sorted(failures)))
    else:
        expected_failures = set()
        with open("expected_failures.txt") as f:
            for l in f.readlines():
                expected_failures.add(l.strip())

        for failure in failures:
            if failure not in expected_failures:
                print("UNEXPECTED FAILURE :(", failure, file=sys.stderr)
                exit_status = 1
        for failure in expected_failures:
            if failure not in failures:
                print("UNEXPECTED PASS :)", failure, file=sys.stderr)
                exit_status = 1
    return exit_status


if __name__ == "__main__":
    sys.exit(main("--update" in sys.argv))
