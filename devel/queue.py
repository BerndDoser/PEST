import multiprocessing
import random
import time


def producer(queue):
    """Task A: Generates data."""
    for i in range(5):
        item = f"Data-Chunk-{i}"
        print(f"PRODUCER: Generating {item}")
        time.sleep(random.uniform(0.1, 0.5))  # Simulate work
        queue.put(item)

    # Send the sentinel signal to shut down the consumer
    queue.put(None)


def consumer(queue):
    """Task B: Processes data."""
    while True:
        item = queue.get()
        if item is None:  # Check for exit signal
            break

        print(f"CONSUMER: Processing {item}...")
        time.sleep(1)  # Simulate a slower downstream task
        print(f"CONSUMER: Finished {item}")


if __name__ == "__main__":
    # The Interface
    pipeline_queue = multiprocessing.Queue()

    # Define Tasks
    p1 = multiprocessing.Process(target=producer, args=(pipeline_queue,))
    c1 = multiprocessing.Process(target=consumer, args=(pipeline_queue,))

    # Start Pipeline
    p1.start()
    c1.start()

    # Wait for completion
    p1.join()
    c1.join()
    print("Pipeline Complete.")
