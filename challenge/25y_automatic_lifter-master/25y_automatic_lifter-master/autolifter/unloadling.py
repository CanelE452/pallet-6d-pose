import asyncio
from forklift_ctrl import AutonomousForkliftController, OperationMode

async def run_job():
    ctrl = AutonomousForkliftController(can_channel=0, can_bitrate=500_000, use_extended_ids=False)
    await ctrl.start()  # 비차단형 내부 루프 시작

    try:
        await ctrl.set_mode(OperationMode.LIFT)
        await ctrl.lift_up(6.0)  # Lifts the forks to a height of 4 units
        await asyncio.sleep(1.0)  # Wait for 1 second to ensure the action is complete

        await ctrl.set_mode(OperationMode.FOLDING)
        await ctrl.fold(0.5)  # Unfolds the forks to the ready position
        await asyncio.sleep(1.0)  # Wait for 1 second

        await ctrl.set_mode(OperationMode.DRIVING)
        await ctrl.drive_forward(50, 9.0)
        await asyncio.sleep(1.0)  # Wait for 1 second

        # Step 2: Unfold the forks to the ready position
        await ctrl.set_mode(OperationMode.FOLDING)
        await ctrl.fold(0.5)  # Unfolds the forks to the ready position
        await asyncio.sleep(1.0)  # Wait for 1 second

                # Step 2: Unfold the forks to the ready position
        await ctrl.set_mode(OperationMode.FOLDING)
        await ctrl.fold(0.5)  # Unfolds the forks to the ready position
        await asyncio.sleep(1.0)  # Wait for 1 second

        # Step 3: Extend the reach mechanism backward
        await ctrl.set_mode(OperationMode.REACH)
        await ctrl.reach_backward(3)  # Extends the reach mechanism backward by 3 units
        await asyncio.sleep(1.0)  # Wait for 1 second

        await ctrl.set_mode(OperationMode.DRIVING)
        await ctrl.drive_backward(50, 14.0)
        await asyncio.sleep(1.0)  # Wait for 1 second
        
        await ctrl.set_mode(OperationMode.LIFT)
        await ctrl.lift_down(5.0)
        await asyncio.sleep(1.0)  # Wait for 1 second

        await ctrl.set_mode(OperationMode.REACH)
        await ctrl.reach_forward(3.5)
        await asyncio.sleep(1.0)  # Wait for 1 second

        await ctrl.set_mode(OperationMode.LIFT)
        await ctrl.lift_down(4.0)
        await asyncio.sleep(1.0)  # Wait for 1 second


        await ctrl.set_mode(OperationMode.FOLDING)
        await ctrl.unfold(1.0)
        await asyncio.sleep(1.0)  # Wait for 1 second
        
        await ctrl.set_mode(OperationMode.LIFT)
        await ctrl.lift_down(0.5)
        await asyncio.sleep(1.0)  # Wait for 1 second
        
        await ctrl.set_mode(OperationMode.DRIVING)
        await ctrl.drive_backward(50, 5.0)
        await asyncio.sleep(1.0)  # Wait for 1 second
    
    finally:
        await ctrl.stop()  # 내부 태스크 종료 + CAN 해제

if __name__ == "__main__":
    asyncio.run(run_job())
