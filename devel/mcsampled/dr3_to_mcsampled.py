"""
Code to produce sampled spectra from Gaia DR3 "continuous" ones using
Rene Andrae's MC sampling.

Source: https://dc.g-vo.org/gaia/s3/ssa/info

@MISC{vo:gdr3spec_SSAP,
  year=2022,
  title={Gaia DR3 MC sampled XP spectra {SSA}},
  author={Demleitner, M. and Andrae, R.},
  url={https://dc.g-vo.org/gaia/s3/ssa/info},
  howpublished={{VO} resource provided by the {GAVO} Data Center},
  doi = {10.21938/W:hWbPah3wBabuqewQULTA}
}
"""

import gzip
import multiprocessing as mp
import os
import pickle
import time

import gaiaxpy
import numpy
import pandas
from scipy import linalg

# where to sample
SPECTRAL_POINTS = numpy.array([400.0 + 10 * i for i in range(41)])

# how many CPUs to keep busy
N_WORKERS = 7

# now many bytes of dump to ~ work on in one go (>~1e7 in operation)
CHUNK_SIZE = 50000000

# how many MC samples to draw for each spectrum
MC_SAMPLES = 20


def _reconstruct_covariance_matrix(coeffs, errs, correlations):
    # Rene's code for producing the covariance matrix.  This should
    # be replaced by gaiaxpy's get_covariance_matrix once we understand
    # what's the difference.
    n = len(coeffs)
    covar = numpy.zeros((n, n))
    correlation_index = 0

    for j in range(n):
        covar[j, j] = errs[j] ** 2
        for i in range(j):
            covar[j, i] = covar[i, j] = correlations[correlation_index] * errs[j] * errs[i]
            correlation_index += 1

    return covar


def get_covariances(rec):
    """returns covariance matrices reconstructed from rec for BP and RP."""
    return _reconstruct_covariance_matrix(
        rec["bp_coefficients"], rec["bp_coefficient_errors"], rec["bp_coefficient_correlations"]
    ), _reconstruct_covariance_matrix(
        rec["rp_coefficients"], rec["rp_coefficient_errors"], rec["rp_coefficient_correlations"]
    )

    # Rene: Do you see why this fails?
    return (
        gaiaxpy.spectrum.utils._get_covariance_matrix(rec, "bp"),
        gaiaxpy.spectrum.utils._get_covariance_matrix(rec, "rp"),
    )


def get_mc_realisations(rec, n_samples):
    """returns n_samples realisations of the continous spectra in rec.

    This assumes that Xp_coefficient_correlations have been replaced
    by the symmetric matrix reconstruction
    (gaiaxpy.core.array_to_symmetric_matrix).
    """
    # We decompose the covariances into L ⋅ LT.  This lets us draw new
    # realisations of the errors as L ⋅ u with a random vector u.
    l_BP = linalg.cholesky(rec["bp_coefficient_correlations"])
    l_RP = linalg.cholesky(rec["rp_coefficient_correlations"])

    samples = []
    coeffs_BP, coeffs_RP = rec["bp_coefficients"], rec["rp_coefficients"]
    length_BP, length_RP = len(coeffs_BP), len(coeffs_RP)
    for s in range(n_samples):
        new_rec = dict(rec)
        new_rec["bp_coefficients"] = coeffs_BP + numpy.dot(l_BP, numpy.random.normal(0, 1, length_BP))
        new_rec["rp_coefficients"] = coeffs_RP + numpy.dot(l_RP, numpy.random.normal(0, 1, length_RP))
        samples.append(new_rec)

    return samples


def cached(func):
    """calls func() and caches the result on disk if it's not already there,
    returns the cache otherwise.
    """
    if os.path.exists("cached.pickle"):
        with open("cached.pickle", "rb") as f:
            return pickle.load(f)

    res = func()
    with open("cached.pickle", "wb") as f:
        pickle.dump(res, f)
    return res


def get_sampled(recs):
    """returns n MCMC-sampled spectrum for a records of a DR3 xp_continuous
    table

    This supports multiple realisations for each source_id and will return
    source_id, flux means per source_id, and the standard deviations within each
    group as errors.
    """
    calib = gaiaxpy.calibrator.calibrator.calibrate(
        pandas.DataFrame.from_records(recs), sampling=SPECTRAL_POINTS, truncation=True, save_file=False
    )

    for name, group in calib[0].groupby("source_id"):
        spec = numpy.mean(group["flux"])
        n = len(group["flux"])
        errs = numpy.sqrt(sum([(s - spec) ** 2 for s in group["flux"]]) / (n - 1))

        yield name, spec, errs


def serialise(source_id, spec, errs):
    """returns a serialised string for a sampled spectrum row."""
    return "{}\t{}\t{}".format(str(source_id), ",".join(f"{v:.7g}" for v in spec), ",".join(f"{v:.7g}" for v in errs))


#######################################################
# Verification/experimentation code


def get_sample_rec(source_id: int = 16011638278912, own_covariances: bool = False):
    """returns some record of gaiadr3.xp_continuous_mean_spectrum.

    This also reconstructs the full covariance matrices in
    Xp_coefficient_correlations
    """
    import pyvo

    srv = pyvo.tap.TAPService("https://gaia.ari.uni-heidelberg.de/tap")
    rec = dict(
        srv.run_sync(f"select * from gaiadr3.xp_continuous_mean_spectrum where source_id={source_id}").to_table()[0]
    )

    if own_covariances:
        rec["bp_coefficient_correlations"], rec["rp_coefficient_correlations"] = get_covariances(rec)
    else:
        from gaiaxpy.core.generic_functions import array_to_symmetric_matrix

        rec["bp_coefficient_correlations"] = array_to_symmetric_matrix(
            rec["bp_coefficient_correlations"], int(rec["bp_n_parameters"])
        )
        rec["rp_coefficient_correlations"] = array_to_symmetric_matrix(
            rec["rp_coefficient_correlations"], int(rec["rp_n_parameters"])
        )

    return rec


def main_verification():
    """a function that produces records with different numbers of MC samples
    and different realisations.

    That's to get an idea of the impact.
    """
    rec = cached(get_sample_rec)
    to_convert = []

    for i in range(5):
        to_convert.extend(get_mc_realisations(rec, 100))
        rec["source_id"] += 1

    for i in range(5):
        to_convert.extend(get_mc_realisations(rec, 50))
        rec["source_id"] += 1

    for i in range(5):
        to_convert.extend(get_mc_realisations(rec, 20))
        rec["source_id"] += 1

    with gzip.open("data3/xp_sampled_computed.txt.gz", mode="wt", encoding="ascii") as out_f:
        for out_rec in get_sampled(to_convert):
            out_f.write(serialise(*out_rec) + "\n")


###########################################################
# Interface for reading from the continuous dump starting

FIELD_NAMES = "source_id,solution_id,bp_basis_function_id,bp_degrees_of_freedom,bp_n_parameters,bp_n_measurements,bp_n_rejected_measurements,bp_standard_deviation,bp_chi_squared,bp_coefficients,bp_coefficient_errors,bp_coefficient_correlations,bp_n_relevant_bases,bp_relative_shrinking,rp_basis_function_id,rp_degrees_of_freedom,rp_n_parameters,rp_n_measurements,rp_n_rejected_measurements,rp_standard_deviation,rp_chi_squared,rp_coefficients,rp_coefficient_errors,rp_coefficient_correlations,rp_n_relevant_bases,rp_relative_shrinking".split(
    ","
)


def _parse_to_records(chunk):
    """returns raw (str-to-str) records from a chunk of lines from the
    database dump.
    """
    for row in chunk.split("\n"):
        if row:
            yield dict(zip(FIELD_NAMES, row.split()))


def _parse_array(arrlit):
    """returns a numpy array from a postgres array literal."""
    return numpy.fromiter((float(literal) for literal in arrlit[1:-1].split(",")), "f")


def _to_types(rec):
    """returns a record with parsed values from a raw record rec.

    This also reconstructs the full covariance matrices in
    Xp_coefficient_correlations
    """
    for arr_col in [
        "bp_chi_squared",
        "bp_coefficients",
        "bp_coefficient_errors",
        "bp_coefficient_correlations",
        "rp_chi_squared",
        "rp_coefficients",
        "rp_coefficient_errors",
        "rp_coefficient_correlations",
    ]:
        rec[arr_col] = _parse_array(rec[arr_col])

    for scalar_col in [
        "bp_standard_deviation",
        "rp_standard_deviation",
    ]:
        rec[scalar_col] = float(rec[scalar_col])

    for int_col in [
        "source_id",
        "bp_n_parameters",
        "rp_n_parameters",
    ]:
        rec[int_col] = int(rec[int_col])

    rec["bp_coefficient_correlations"], rec["rp_coefficient_correlations"] = get_covariances(rec)

    return rec


def _iter_records(src_name):
    """yields parsed records from a gzipped dump of the dr3 continous
    spectrum table.
    """
    with gzip.open(src_name, "rt") as f:
        for raw in _parse_to_records(f.read()):
            yield _to_types(raw)


##################################################3
# Dump conversion including parallel workers


class Worker:
    # the controller-side representation of a processing job
    def __init__(self, id):
        self.id = id
        self.in_queue = mp.Queue()
        self.out_queue = mp.Queue()
        self._process = mp.Process(target=self._run, args=(self.id, self.in_queue, self.out_queue))
        self._process.start()
        self.chunks_not_returned = 0
        self.joined = False

    def submit(self, data):
        self.in_queue.put(data)
        if data:
            self.chunks_not_returned += 1

    @staticmethod
    def _run(id, in_queue, out_queue):
        while True:
            chunk = in_queue.get()
            if not chunk:
                # we're done; exit
                break

            buffer = []
            for row in _parse_to_records(chunk):
                buffer.extend(get_mc_realisations(_to_types(row), MC_SAMPLES))

            serialised = []
            for source_id, flux, flux_error in get_sampled(buffer):
                serialised.append(serialise(source_id, flux, flux_error))

            out_queue.put("\n".join(serialised))
        print(f"Process {id} exiting.")

    def write_result(self, dest_f):
        """will write a result to dest_f if one is there.

        This is a no-op if the worker has no new results.
        """
        if not self.out_queue.empty():
            print(f"Worker {self.id} writing result")
            dest_f.write(self.out_queue.get() + "\n")
            self.chunks_not_returned -= 1

    def join(self):
        self._process.join(1)
        if self._process.exitcode is not None:
            self.joined = True


def main():
    workers = [Worker(str(i)) for i in range(N_WORKERS)]

    with (
        gzip.open("data3/xp_sampled_computed.txt.gz", mode="wt", encoding="ascii") as out_f,
        gzip.open("data3/xp_continuous_dump.txt.gz", mode="rt", encoding="ascii") as in_f,
    ):
        chunk = in_f.read(CHUNK_SIZE)
        line_sep = -1
        while chunk:
            rest = in_f.read(CHUNK_SIZE)
            if len(rest) < CHUNK_SIZE:
                # file exhausted, let the last job do its thing and exit
                chunk = chunk + rest
                rest = ""

            else:
                # normal operation: complete last line of chunk, hand off packet.
                line_sep = rest.index("\n")
                chunk = chunk + rest[:line_sep]

            # Wait for the next worker to be finish and submit the next chunk to
            # them.
            while True:
                # Let workers write any results coming in
                for w in workers:
                    w.write_result(out_f)

                for w in workers:
                    if w.chunks_not_returned == 0:
                        print(f"Submitting to worker {w.id}")
                        w.submit(chunk)
                        chunk, rest = rest[line_sep + 1 :], ""
                        break
                else:  # no free worker.  Retry in a bit
                    time.sleep(2)
                    continue
                break

        # tell workers to exit
        print("Asking Workers to exit")
        for w in workers:
            w.submit("")

        # when exiting the main loop, make sure all workers have finished
        for retry in range(100):
            print("Waiting for " + ", ".join(w.id for w in workers))
            not_joined_yet = []
            for w in workers:
                w.write_result(out_f)
                if w.chunks_not_returned == 0:
                    w.join()
                else:
                    not_joined_yet.append(w)

            workers = not_joined_yet
            if not workers:
                break
            time.sleep(2)
        else:
            raise Exception("Hung workers?")


if __name__ == "__main__":
    main()
