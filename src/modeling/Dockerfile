# declare base image
FROM public.ecr.aws/lambda/python:3.10

# install dependencies
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# copy function code
COPY main.py ${LAMBDA_TASK_ROOT}
COPY extract_clean_data.py ${LAMBDA_TASK_ROOT}
COPY score_model.py ${LAMBDA_TASK_ROOT}
COPY evaluate_model.py ${LAMBDA_TASK_ROOT}
COPY train_model.py ${LAMBDA_TASK_ROOT}
COPY tune_model.py ${LAMBDA_TASK_ROOT}
COPY upload_artifacts.py ${LAMBDA_TASK_ROOT}
COPY config/local.conf ${LAMBDA_TASK_ROOT}

# set CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD ["main.lambda_handler"]