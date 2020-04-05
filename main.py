import sys
import ast

from job.points_of_interest_job import PointOfInterest
from job.points_of_interest_validation_job import PointsOfInterestValidation
from foundation.configuration.input import Input

def start_input(args):
    Input().set_inputs(args)


def start_job(args):

    start_input(args)
    job_name = Input.get_instance().inputs['job']
    print(job_name)
    if job_name == "points_of_interest_job":
        job = PointOfInterest()
    elif job_name == "points_of_interest_validation_job":
        job = PointsOfInterestValidation()

    job.start()

if __name__ == "__main__":
    try:

        args = ast.literal_eval(sys.argv[1])
        start_job(args)

    except Exception as e:
        raise e