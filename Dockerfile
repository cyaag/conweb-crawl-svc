# Use the AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.10

# Set the working directory in the container
# WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ${LAMBDA_TASK_ROOT} 

# Install any dependencies specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY app.py ${LAMBDA_TASK_ROOT} 

# Command to run the Lambda function
CMD ["app.lambda_handler"]