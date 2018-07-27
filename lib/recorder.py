from gym.wrappers import Monitor

class Recorder(Monitor):
    def __init__(self,env,directory,video_callable=None, force=False, resume=False,
                 write_upon_reset=False, uid=None, mode=None):
        super().__init__(env,directory,video_callable=video_callable, force=force, resume=resume,
                 write_upon_reset=write_upon_reset, uid=uid, mode=mode)
        self.car = self.env.car
        self.driver = self.env.driver
        self.state = self.env.state
        # self.leading_car = self.env.leading_car
    def update_state(self,action):
        return self.env.update_state(action)
    
    def calculate_the_reward(self,action,state_memory,index_next_state):
        return self.env.calculate_the_reward(self,action,state_memory,index_next_state)

    
    


    






