import subprocess
import sys
import os
import time
import signal
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

@dataclass
class DebugProcess:
    pid: int
    host: str
    port: int
    process: subprocess.Popen
    entry_point: str
    start_time: float

class DebugRuntime:
    """디버그 런타임 관리"""
    
    def __init__(self, default_host: str = "127.0.0.1", default_port: int = 5678):
        self.default_host = default_host
        self.default_port = default_port
        self.active_processes: Dict[int, DebugProcess] = {}
        self.port_allocator = PortAllocator(default_port)
    
    def run_with_debugpy(
        self,
        entry: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        args: Optional[List[str]] = None
    ) -> DebugProcess:
        """debugpy와 함께 프로세스 실행"""
        
        host = host or self.default_host
        port = port or self.port_allocator.get_next_available_port()
        
        # 환경 변수 설정
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        
        # debugpy 설정
        debugpy_code = f"""
import debugpy
import sys
import os

# debugpy 설정
debugpy.listen(('{host}', {port}))
print(f'debugpy attached on {host}:{port}')
print('Waiting for debugger to attach...')

# 클라이언트 연결 대기 (선택사항)
# debugpy.wait_for_client()

# 엔트리 포인트 실행
try:
    if '{entry}'.endswith('.py'):
        import runpy
        runpy.run_path('{entry}')
    else:
        # 모듈로 실행
        import importlib.util
        spec = importlib.util.spec_from_file_location('main', '{entry}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
except Exception as e:
    print(f'Error running {entry}: {{e}}')
    import traceback
    traceback.print_exc()
"""
        
        # 명령어 구성
        cmd = [sys.executable, "-c", debugpy_code]
        if args:
            cmd.extend(args)
        
        try:
            # 프로세스 시작
            process = subprocess.Popen(
                cmd,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd()
            )
            
            # 프로세스 정보 저장
            debug_process = DebugProcess(
                pid=process.pid,
                host=host,
                port=port,
                process=process,
                entry_point=entry,
                start_time=time.time()
            )
            
            self.active_processes[process.pid] = debug_process
            
            # 포트 할당
            self.port_allocator.allocate_port(port)
            
            return debug_process
            
        except Exception as e:
            raise RuntimeError(f"Failed to start debug process: {str(e)}")
    
    def run_with_debugpy_module(
        self,
        module_name: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        args: Optional[List[str]] = None
    ) -> DebugProcess:
        """모듈로 debugpy와 함께 실행"""
        
        host = host or self.default_host
        port = port or self.port_allocator.get_next_available_port()
        
        # 환경 변수 설정
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        
        # debugpy 설정
        debugpy_code = f"""
import debugpy
import sys
import os

# debugpy 설정
debugpy.listen(('{host}', {port}))
print(f'debugpy attached on {host}:{port}')
print('Waiting for debugger to attach...')

# 클라이언트 연결 대기 (선택사항)
# debugpy.wait_for_client()

# 모듈 실행
try:
    import runpy
    runpy.run_module('{module_name}', run_name='__main__')
except Exception as e:
    print(f'Error running module {module_name}: {{e}}')
    import traceback
    traceback.print_exc()
"""
        
        # 명령어 구성
        cmd = [sys.executable, "-c", debugpy_code]
        if args:
            cmd.extend(args)
        
        try:
            # 프로세스 시작
            process = subprocess.Popen(
                cmd,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd()
            )
            
            # 프로세스 정보 저장
            debug_process = DebugProcess(
                pid=process.pid,
                host=host,
                port=port,
                process=process,
                entry_point=module_name,
                start_time=time.time()
            )
            
            self.active_processes[process.pid] = debug_process
            
            # 포트 할당
            self.port_allocator.allocate_port(port)
            
            return debug_process
            
        except Exception as e:
            raise RuntimeError(f"Failed to start debug process: {str(e)}")
    
    def stop_process(self, pid: int) -> bool:
        """프로세스 중지"""
        if pid not in self.active_processes:
            return False
        
        debug_process = self.active_processes[pid]
        
        try:
            # 프로세스 종료
            debug_process.process.terminate()
            
            # 5초 대기 후 강제 종료
            try:
                debug_process.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                debug_process.process.kill()
            
            # 포트 해제
            self.port_allocator.release_port(debug_process.port)
            
            # 프로세스 정보 제거
            del self.active_processes[pid]
            
            return True
            
        except Exception as e:
            print(f"Error stopping process {pid}: {str(e)}")
            return False
    
    def stop_all_processes(self) -> int:
        """모든 프로세스 중지"""
        stopped_count = 0
        
        for pid in list(self.active_processes.keys()):
            if self.stop_process(pid):
                stopped_count += 1
        
        return stopped_count
    
    def get_process_info(self, pid: int) -> Optional[Dict[str, Any]]:
        """프로세스 정보 조회"""
        if pid not in self.active_processes:
            return None
        
        debug_process = self.active_processes[pid]
        
        # 프로세스 상태 확인
        return_code = debug_process.process.poll()
        status = "running" if return_code is None else f"exited({return_code})"
        
        return {
            "pid": pid,
            "host": debug_process.host,
            "port": debug_process.port,
            "entry_point": debug_process.entry_point,
            "status": status,
            "start_time": debug_process.start_time,
            "uptime": time.time() - debug_process.start_time,
            "return_code": return_code
        }
    
    def list_processes(self) -> List[Dict[str, Any]]:
        """활성 프로세스 목록"""
        return [self.get_process_info(pid) for pid in self.active_processes.keys()]
    
    def cleanup_dead_processes(self) -> int:
        """죽은 프로세스 정리"""
        cleaned_count = 0
        
        for pid in list(self.active_processes.keys()):
            debug_process = self.active_processes[pid]
            
            if debug_process.process.poll() is not None:
                # 프로세스가 종료됨
                self.port_allocator.release_port(debug_process.port)
                del self.active_processes[pid]
                cleaned_count += 1
        
        return cleaned_count

class PortAllocator:
    """포트 할당 관리"""
    
    def __init__(self, start_port: int = 5678, max_ports: int = 100):
        self.start_port = start_port
        self.max_ports = max_ports
        self.allocated_ports: set = set()
    
    def get_next_available_port(self) -> int:
        """사용 가능한 다음 포트 찾기"""
        for port in range(self.start_port, self.start_port + self.max_ports):
            if port not in self.allocated_ports:
                return port
        
        raise RuntimeError("No available ports")
    
    def allocate_port(self, port: int) -> bool:
        """포트 할당"""
        if port in self.allocated_ports:
            return False
        
        self.allocated_ports.add(port)
        return True
    
    def release_port(self, port: int) -> bool:
        """포트 해제"""
        if port in self.allocated_ports:
            self.allocated_ports.remove(port)
            return True
        
        return False
    
    def is_port_available(self, port: int) -> bool:
        """포트 사용 가능 여부 확인"""
        return port not in self.allocated_ports

# 편의 함수들
def run_with_debugpy_simple(
    entry: str,
    host: str = "127.0.0.1",
    port: int = 5678
) -> subprocess.Popen:
    """간단한 debugpy 실행 (기존 함수와의 호환성)"""
    runtime = DebugRuntime()
    debug_process = runtime.run_with_debugpy(entry, host, port)
    return debug_process.process

def run_with_debugpy_module_simple(
    module_name: str,
    host: str = "127.0.0.1",
    port: int = 5678
) -> subprocess.Popen:
    """간단한 모듈 debugpy 실행"""
    runtime = DebugRuntime()
    debug_process = runtime.run_with_debugpy_module(module_name, host, port)
    return debug_process.process

# 전역 디버그 런타임 인스턴스
_global_runtime = DebugRuntime()

def get_global_runtime() -> DebugRuntime:
    """전역 디버그 런타임 인스턴스 반환"""
    return _global_runtime

