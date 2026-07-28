import asyncio
from forklift_ctrl import AutonomousForkliftController, OperationMode

async def run_job():
    ctrl = AutonomousForkliftController(can_channel=0, can_bitrate=500_000, use_extended_ids=False)
    await ctrl.start()  # 비차단형 내부 루프 시작

    try:
       
        await ctrl.set_mode(OperationMode.LIFT)
        await ctrl.lift_up(6.0)
        await asyncio.sleep(1.0)  # Wait for 1 second to ensure the action is complete  

        await ctrl.set_mode(OperationMode.REACH)
        await ctrl.reach_forward(3)
        await asyncio.sleep(1.0)  # Wait for 1 second to ensure the action is complete  

        await ctrl.set_mode(OperationMode.DRIVING)
        await ctrl.drive_forward(50, 9.0)
        await asyncio.sleep(1.0)

        await ctrl.set_mode(OperationMode.FOLDING)
        await ctrl.unfold(0.5)
        await asyncio.sleep(1.0)  # Wait for 1 second to ensure the action is complete  
       
        await ctrl.set_mode(OperationMode.FOLDING)
        await ctrl.unfold(0.5)
        await asyncio.sleep(1.0)  # Wait for 1 second to ensure the action is complete  

        await ctrl.set_mode(OperationMode.LIFT)
        await ctrl.lift_down(1.0)
        await asyncio.sleep(1.0)  # Wait for 1 second to ensure the action is complete  

        await ctrl.set_mode(OperationMode.DRIVING)
        await ctrl.drive_backward(50, 12.0)
        await asyncio.sleep(1.0)  # Wait for 1 second to ensure the action is complete  

        await ctrl.set_mode(OperationMode.LIFT)
        await ctrl.lift_down(6.0)
        await asyncio.sleep(1.0)  # Wait for 1 second to ensure the action is complete  


    finally:
        await ctrl.stop()  # 내부 태스크 종료 + CAN 해제

if __name__ == "__main__":
    asyncio.run(run_job())
