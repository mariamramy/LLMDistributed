import argparse
import multiprocessing
import time
import logging
from master.scheduler import MasterScheduler
from lb.load_balancer import LoadBalancer, Node, STRATEGIES
from client.load_generator import run_load_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MAIN] %(levelname)s %(message)s",
)
log = logging.getLogger("main")


# Configuration
MASTER_HOST = "127.0.0.1"
MASTER_PORT = 9000

LB_HOST = "127.0.0.1"
LB_PORT = 8080

STARTUP_WAIT = 2.0  # seconds to let services start before sending traffic



# Process target functions
def start_master(host: str, port: int):
    scheduler = MasterScheduler(host=host, port=port)
    scheduler.run()


def start_lb(host: str, port: int, strategy_name: str, nodes):
    lb = LoadBalancer(
        nodes=nodes,
        strategy=STRATEGIES[strategy_name],
        host=host,
        port=port,
    )
    lb.run()


def main():
    parser = argparse.ArgumentParser(description="Distributed LLM Cluster — Main Entry")
    parser.add_argument("--users",    type=int, default=100,          help="Number of simulated concurrent users")
    parser.add_argument("--rpu",      type=int, default=1,            help="Requests per user")
    parser.add_argument("--strategy", default="round_robin",
                        choices=list(STRATEGIES.keys()),              help="Load balancing strategy")
    args = parser.parse_args()


    # 1. Start Master Scheduler
    master_proc = multiprocessing.Process(
        target=start_master,
        args=(MASTER_HOST, MASTER_PORT),
        daemon=True,
        name="MasterScheduler",
    )
    master_proc.start()
    log.info("Master Scheduler process started (pid=%d)", master_proc.pid)

    # 2. Start Load Balancer 
    backend_nodes = [
        Node(node_id="master-1", host=MASTER_HOST, port=MASTER_PORT),
    ]

    lb_proc = multiprocessing.Process(
        target=start_lb,
        args=(LB_HOST, LB_PORT, args.strategy, backend_nodes),
        daemon=True,
        name="LoadBalancer",
    )
    lb_proc.start()
    log.info("Load Balancer process started (pid=%d)", lb_proc.pid)


    # 3. Wait for services to initialise
    log.info("Waiting %.1fs for services to start…", STARTUP_WAIT)
    time.sleep(STARTUP_WAIT)


    # 4. Run Client Load Test
    log.info(
        "Starting load test: %d users × %d request(s) each | strategy=%s",
        args.users, args.rpu, args.strategy,
    )
    run_load_test(
        num_users=args.users,
        requests_per_user=args.rpu,
        lb_host=LB_HOST,
        lb_port=LB_PORT,
    )

    
    # 5. Teardown
    log.info("Load test complete — shutting down services.")
    master_proc.terminate()
    lb_proc.terminate()
    master_proc.join(timeout=3)
    lb_proc.join(timeout=3)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
