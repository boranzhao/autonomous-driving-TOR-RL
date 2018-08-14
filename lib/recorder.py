from gym.wrappers import Monitor

class Recorder(Monitor):
    def __init__(self,env,directory,video_callable=None, force=False, resume=False,
                 write_upon_reset=False, uid=None, mode=None):
        super().__init__(env,directory,video_callable=video_callable, force=force, resume=resume,
                 write_upon_reset=write_upon_reset, uid=uid, mode=mode)
        self.car = self.env.car
        self.driver = self.env.driver
        self.state = self.env.state
    def step(self, action):
        # self._before_step(action)    # This line is commented 
        observation, reward, done, info = self.env.step(action)
        done = self._after_step(observation, reward, done, info)
        # update the stats
        self.num_near_crashes  = self.env.num_near_crashes
        self.num_crashes = self.env.num_crashes
        
        return observation, reward, done, info

    def _after_step(self, observation, reward, done, info):
        if not self.enabled: return done

        if done and self.env_semantics_autoreset:
            # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
            self.reset_video_recorder()
            self.episode_id += 1
            self._flush()

        # # Record stats
        # self.stats_recorder.after_step(observation, reward, done, info)
        # Record video
        self.video_recorder.capture_frame()

        return done

    # def step(self,action):
    #     return self.env.step(action)
    

    


    






