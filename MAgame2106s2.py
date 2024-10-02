
# partner selection, PD
# Q, knowing every opponent past action

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from Environment02 import Environment02

from Agent36b import Agent36b

def mainTrain(gameName, Nact, roundsG, algoName, lr, Nagent, simStart, Nsim, tStart, T):

    Npast = 1
    dirName1 = 'result_%s_%s_lr%.2f_Npast%d_Nagent%d_R%d'%(gameName, algoName, lr, Npast, Nagent, roundsG)
    print(dirName1)
    if not os.path.exists(dirName1):
        os.makedirs(dirName1)

    reward1s = {}
    reward2s = {}
    # PD2
    reward1s['PD2'] = np.array([[3, 0], [5, 1]])
    reward2s['PD2'] = np.array([[3, 5], [0, 1]])

    env = Environment02(Nact, reward1s[gameName], reward2s[gameName])

    np.random.seed()

    for sim in range(simStart, Nsim+1):
        dirName = '%s/sim%04d'%(dirName1,sim)
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        Nelement1 = (Npast+1)*2 + (Npast+1)*Nact
        StatQmeanT = np.zeros((T + 1, Nelement1), dtype=np.float32)
        StatQvarT = np.zeros((T + 1, Nelement1), dtype=np.float32)
        StatPmeanT = np.zeros((T + 1, Nelement1), dtype=np.float32)
        StatPvarT = np.zeros((T + 1, Nelement1), dtype=np.float32)

        CountSwitchT = np.zeros((T, 2), dtype=int) # 0:N; 1:Y (isSwitch)
        CountOutcomeT = np.zeros((T, 4), dtype=int) # 0:(C,C); 1:(C,D); 2:(D,C); 3:(D,D)
        CountActualOutcomeT = np.zeros((T, 8), dtype=int) #  0:Stay(C,C); 1:Stay(C,D); 2:Stay(D,C); 3:Stay(D,D)
                                                                #  4:Switch(C,C); 5:Switch(C,D); 6:Switch(D,C); 7:Switch(D,D)
        AgentRewardSumT = np.zeros((T, Nagent), dtype=int)

        AgentPolicySwTypeT = np.zeros((T, Nagent), dtype=int)  # 0; 1; 2; 3
        AgentPolicyPDTypeT = np.zeros((T, Nagent), dtype=int) # 0; 1; 2; 3
        CountPolicySwTypeT = np.zeros((T, 4), dtype=int)  # N|oC,N|oD; N|oC,Y|oD; Y|oC,N|oD; Y|oC,Y|oD
        CountPolicyPDTypeT = np.zeros((T, 4), dtype=int)  # C|oC,C|oD; C|oC,D|oD; D|oC,C|oD; D|oC,D|oD

        AgentCountSwT = np.zeros((T, Nagent), dtype=int)  # 0-roundG, number of Y (switch)
        AgentCountDT = np.zeros((T, Nagent), dtype=int)  # 0-roundG, number of D

        CountSwitch_oAT = np.zeros((T, 4), dtype=int) # 0:N|oC; 1:Y|oC; 2:N|oD; 3:Y|oD;
        CountCD_oAT = np.zeros((T, 4), dtype=int) # 0:C|oC; 1:D|oC; 2:C|oD; 3:D|oD;

        if tStart != 0:
            StatQmeanTinit = np.loadtxt(('%s/StatQmeanT-sim%04d.txt')%(dirName, sim), delimiter=',')
            StatQvarTinit = np.loadtxt(('%s/StatQvarT-sim%04d.txt')%(dirName, sim), delimiter=',')
            StatPmeanTinit = np.loadtxt(('%s/StatPmeanT-sim%04d.txt')%(dirName, sim), delimiter=',')
            StatPvarTinit = np.loadtxt(('%s/StatPvarT-sim%04d.txt')%(dirName, sim), delimiter=',')
            StatQmeanT[:tStart+1] = StatQmeanTinit[:tStart+1]
            StatQvarT[:tStart+1] = StatQvarTinit[:tStart+1]
            StatPmeanT[:tStart+1] = StatPmeanTinit[:tStart+1]
            StatPvarT[:tStart+1] = StatPvarTinit[:tStart+1]
            CountSwitchTinit = np.loadtxt(('%s/CountSwitchT-sim%04d.txt')%(dirName, sim), delimiter=',')
            CountOutcomeTinit = np.loadtxt(('%s/CountOutcomeT-sim%04d.txt') % (dirName, sim), delimiter=',')
            AgentRewardSumTinit = np.loadtxt(('%s/AgentRewardSumT-sim%04d.txt') % (dirName, sim), delimiter=',')
            CountSwitchT[:tStart] = CountSwitchTinit[:tStart]
            CountOutcomeT[:tStart] = CountOutcomeTinit[:tStart]
            AgentRewardSumT[:tStart] = AgentRewardSumTinit[:tStart]
            CountActualOutcomeTinit = np.loadtxt(('%s/CountActualOutcomeT-sim%04d.txt') % (dirName, sim), delimiter=',')
            CountActualOutcomeT[:tStart] = CountActualOutcomeTinit[:tStart]
            AgentPolicySwTypeTinit = np.loadtxt(('%s/AgentPolicySwTypeT-sim%04d.txt') % (dirName, sim), delimiter=',')
            CountPolicySwTypeTinit = np.loadtxt(('%s/CountPolicySwTypeT-sim%04d.txt') % (dirName, sim), delimiter=',')
            AgentPolicySwTypeT[:tStart] = AgentPolicySwTypeTinit[:tStart]
            CountPolicySwTypeT[:tStart] = CountPolicySwTypeTinit[:tStart]
            AgentPolicyPDTypeTinit = np.loadtxt(('%s/AgentPolicyPDTypeT-sim%04d.txt') % (dirName, sim), delimiter=',')
            CountPolicyPDTypeTinit = np.loadtxt(('%s/CountPolicyPDTypeT-sim%04d.txt') % (dirName, sim), delimiter=',')
            AgentPolicyPDTypeT[:tStart] = AgentPolicyPDTypeTinit[:tStart]
            CountPolicyPDTypeT[:tStart] = CountPolicyPDTypeTinit[:tStart]
            AgentCountSwTinit = np.loadtxt(('%s/AgentCountSwT-sim%04d.txt') % (dirName, sim), delimiter=',')
            AgentCountDTinit = np.loadtxt(('%s/AgentCountDT-sim%04d.txt') % (dirName, sim), delimiter=',')
            AgentCountSwT[:tStart] = AgentCountSwTinit[:tStart]
            AgentCountDT[:tStart] = AgentCountDTinit[:tStart]
            CountSwitch_oATinit = np.loadtxt(('%s/CountSwitch_oAT-sim%04d.txt') % (dirName, sim), delimiter=',')
            CountCD_oATinit = np.loadtxt(('%s/CountCD_oAT-sim%04d.txt') % (dirName, sim), delimiter=',')
            CountSwitch_oAT[:tStart] = CountSwitch_oATinit[:tStart]
            CountCD_oAT[:tStart] = CountCD_oATinit[:tStart]
        else:
            StatPmeanT[0, :] = 1/2

        # initialize
        if tStart != 0:
            filePath1 = '%s/Agents-sim%04d_T%d.pickle' % (dirName, sim, tStart)
            f01 = open(filePath1, 'rb')
            agents = pickle.load(f01)
            f01.close()
            filePath1 = '%s/Groups-sim%04d_T%d.pickle' % (dirName, sim, tStart)
            f01 = open(filePath1, 'rb')
            groups = pickle.load(f01)
            f01.close()
            filePath1 = '%s/ActionsPD-sim%04d_T%d.pickle' % (dirName, sim, tStart)
            f01 = open(filePath1, 'rb')
            actionsPD = pickle.load(f01)
            f01.close()
        else:
            agents = []
            for i in range(Nagent):
                agent = Agent36b(env, lr=lr, name='%s_%d'%(algoName, i))
                agents.append(agent)
            groups = np.arange(Nagent)
            np.random.shuffle(groups)
            groups = groups.reshape((-1, 2))
            actionsPD = np.random.choice(Nact, Nagent)

        for t in range(tStart+1, T+1):
            CountSwitch = np.zeros(2, dtype=int)
            CountOutcome = np.zeros(4, dtype=int)
            CountActualOutcome = np.zeros(8, dtype=int)
            AgentRewardSum = np.zeros(Nagent, dtype=int)
            AgentCountSw = np.zeros(Nagent, dtype=int)
            AgentCountD = np.zeros(Nagent, dtype=int)
            CountSwitch_oA = np.zeros(4, dtype=int)
            CountCD_oA = np.zeros(4, dtype=int)
            for round1 in range(roundsG):
                groupsStay = np.zeros((0, 2), dtype=int)
                groupsSwitch = np.array([], dtype=int)
                isSwitch = np.zeros(Nagent, dtype=int)
                # isSwitch
                for group in groups:
                    i = group[0]
                    j = group[1]
                    s = [actionsPD[j], actionsPD[i]]
                    isSwitch1 = [agents[i].getAction(s[0]), agents[j].getAction(s[1])]
                    agents[i].storeMemory(s[0], isSwitch1[0], 0)
                    agents[j].storeMemory(s[1], isSwitch1[1], 0)
                    CountSwitch[isSwitch1[0]] += 1
                    CountSwitch[isSwitch1[1]] += 1
                    CountSwitch_oA[s[0]*2 + isSwitch1[0]] += 1
                    CountSwitch_oA[s[1]*2 + isSwitch1[1]] += 1
                    AgentCountSw[i] += isSwitch1[0]
                    AgentCountSw[j] += isSwitch1[1]

                    if isSwitch1[0] == 1 or isSwitch1[1] == 1:
                        groupsSwitch = np.concatenate((groupsSwitch, group))
                        isSwitch[i] = 1
                        isSwitch[j] = 1
                    else:
                        groupsStay = np.concatenate((groupsStay, group[np.newaxis, :]), axis=0)
                np.random.shuffle(groupsSwitch)
                groupsSwitch = groupsSwitch.reshape((-1, 2))
                groups = np.concatenate((groupsStay, groupsSwitch), axis=0)

                # PD
                for group in groups:
                    i = group[0]
                    j = group[1]
                    s = [100 + actionsPD[j], 100 + actionsPD[i]]
                    actionsPD1 = [agents[i].getAction(s[0]), agents[j].getAction(s[1])]
                    rewards1 = env.getRewards(actionsPD1)
                    agents[i].storeMemory(s[0], actionsPD1[0], rewards1[0])
                    agents[j].storeMemory(s[1], actionsPD1[1], rewards1[1])
                    CountOutcome[2 * actionsPD1[0] + actionsPD1[1]] += 1
                    CountActualOutcome[4*isSwitch[i] + 2*actionsPD1[0] + actionsPD1[1]] += 1
                    CountCD_oA[(s[0]-100)*2 + actionsPD1[0]] += 1
                    CountCD_oA[(s[1]-100)*2 + actionsPD1[1]] += 1
                    AgentRewardSum[i] += rewards1[0]
                    AgentRewardSum[j] += rewards1[1]
                    AgentCountD[i] += actionsPD1[0]
                    AgentCountD[j] += actionsPD1[1]

                    actionsPD[i] = actionsPD1[0]
                    actionsPD[j] = actionsPD1[1]
            for i in range(Nagent):
                agents[i].train()

            # record
            Qsum = np.zeros(Nelement1, dtype=np.float32)
            Qsqsum = np.zeros(Nelement1, dtype=np.float32)
            Psum = np.zeros(Nelement1, dtype=np.float32)
            Psqsum = np.zeros(Nelement1, dtype=np.float32)
            AgentPolicySwType = np.zeros(Nagent, dtype=int)
            CountPolicySwType = np.zeros(4, dtype=int)
            AgentPolicyPDType = np.zeros(Nagent, dtype=int)
            CountPolicyPDType = np.zeros(4, dtype=int)
            for i in range(len(agents)):
                Qsum += agents[i].getQall()
                Qsqsum += agents[i].getQall() ** 2
                Psum += agents[i].getPolicyAll()
                Psqsum += agents[i].getPolicyAll() ** 2

                Q_oC = agents[i].Q[0]
                Q_oD = agents[i].Q[1]
                if Q_oC[0] > Q_oC[1] and Q_oD[0] > Q_oD[1]:
                    CountPolicySwType[0] += 1
                    AgentPolicySwType[i] = 0
                elif Q_oC[0] > Q_oC[1] and Q_oD[0] <= Q_oD[1]:
                    CountPolicySwType[1] += 1
                    AgentPolicySwType[i] = 1
                elif Q_oC[0] <= Q_oC[1] and Q_oD[0] > Q_oD[1]:
                    CountPolicySwType[2] += 1
                    AgentPolicySwType[i] = 2
                else:
                    CountPolicySwType[3] += 1
                    AgentPolicySwType[i] = 3

                Q_oC = agents[i].Q[100]
                Q_oD = agents[i].Q[101]
                if Q_oC[0] > Q_oC[1] and Q_oD[0] > Q_oD[1]:
                    CountPolicyPDType[0] += 1
                    AgentPolicyPDType[i] = 0
                elif Q_oC[0] > Q_oC[1] and Q_oD[0] <= Q_oD[1]:
                    CountPolicyPDType[1] += 1
                    AgentPolicyPDType[i] = 1
                elif Q_oC[0] <= Q_oC[1] and Q_oD[0] > Q_oD[1]:
                    CountPolicyPDType[2] += 1
                    AgentPolicyPDType[i] = 2
                else:
                    CountPolicyPDType[3] += 1
                    AgentPolicyPDType[i] = 3
            StatQmeanT[t] = Qsum / Nagent
            StatQvarT[t] = Qsqsum / Nagent - (Qsum / Nagent) ** 2
            StatPmeanT[t] = Psum / Nagent
            StatPvarT[t] = Psqsum / Nagent - (Psum / Nagent) ** 2
            CountSwitchT[t-1] = CountSwitch
            CountOutcomeT[t-1] = CountOutcome
            CountActualOutcomeT[t - 1] = CountActualOutcome
            AgentRewardSumT[t-1] = AgentRewardSum
            AgentCountSwT[t - 1] = AgentCountSw
            AgentCountDT[t - 1] = AgentCountD

            AgentPolicySwTypeT[t-1] = AgentPolicySwType
            CountPolicySwTypeT[t-1] = CountPolicySwType
            AgentPolicyPDTypeT[t - 1] = AgentPolicyPDType
            CountPolicyPDTypeT[t - 1] = CountPolicyPDType
            CountSwitch_oAT[t-1] = CountSwitch_oA
            CountCD_oAT[t-1] = CountCD_oA

            if t % 1000 == 0 or t==T:
                filePath1 = '%s/Agents-sim%04d_T%d.pickle' % (dirName, sim, t)
                f01 = open(filePath1, 'wb')
                pickle.dump(agents, f01)
                f01.close()
                filePath1 = '%s/Groups-sim%04d_T%d.pickle' % (dirName, sim, t)
                f01 = open(filePath1, 'wb')
                pickle.dump(groups, f01)
                f01.close()
                filePath1 = '%s/ActionsPD-sim%04d_T%d.pickle' % (dirName, sim, t)
                f01 = open(filePath1, 'wb')
                pickle.dump(actionsPD, f01)
                f01.close()

                np.savetxt(('%s/StatQmeanT-sim%04d.txt') % (dirName, sim), StatQmeanT, fmt='%.6f', delimiter=',')
                np.savetxt(('%s/StatQvarT-sim%04d.txt') % (dirName, sim), StatQvarT, fmt='%.6f', delimiter=',')
                np.savetxt(('%s/StatPmeanT-sim%04d.txt') % (dirName, sim), StatPmeanT, fmt='%.6f', delimiter=',')
                np.savetxt(('%s/StatPvarT-sim%04d.txt') % (dirName, sim), StatPvarT, fmt='%.6f', delimiter=',')
                np.savetxt(('%s/CountSwitchT-sim%04d.txt') % (dirName, sim), CountSwitchT, fmt='%d', delimiter=',')
                np.savetxt(('%s/CountOutcomeT-sim%04d.txt') % (dirName, sim), CountOutcomeT, fmt='%d', delimiter=',')
                np.savetxt(('%s/CountActualOutcomeT-sim%04d.txt') % (dirName, sim), CountActualOutcomeT, fmt='%d', delimiter=',')
                np.savetxt(('%s/AgentRewardSumT-sim%04d.txt') % (dirName, sim), AgentRewardSumT, fmt='%d', delimiter=',')
                np.savetxt(('%s/AgentCountSwT-sim%04d.txt') % (dirName, sim), AgentCountSwT, fmt='%d', delimiter=',')
                np.savetxt(('%s/AgentCountDT-sim%04d.txt') % (dirName, sim), AgentCountDT, fmt='%d', delimiter=',')
                np.savetxt(('%s/AgentPolicySwTypeT-sim%04d.txt') % (dirName, sim), AgentPolicySwTypeT, fmt='%d', delimiter=',')
                np.savetxt(('%s/CountPolicySwTypeT-sim%04d.txt') % (dirName, sim), CountPolicySwTypeT, fmt='%d', delimiter=',')
                np.savetxt(('%s/AgentPolicyPDTypeT-sim%04d.txt') % (dirName, sim), AgentPolicyPDTypeT, fmt='%d', delimiter=',')
                np.savetxt(('%s/CountPolicyPDTypeT-sim%04d.txt') % (dirName, sim), CountPolicyPDTypeT, fmt='%d', delimiter=',')
                np.savetxt(('%s/CountSwitch_oAT-sim%04d.txt') % (dirName, sim), CountSwitch_oAT, fmt='%d', delimiter=',')
                np.savetxt(('%s/CountCD_oAT-sim%04d.txt') % (dirName, sim), CountCD_oAT, fmt='%d', delimiter=',')

                print('sim %d time %d' % (sim, t), CountPolicySwTypeT[t-1], CountPolicyPDTypeT[t-1], CountActualOutcomeT[t-1])
        print('sim %d completed'%(sim))
    

    percentages = np.zeros((T, 4))

    for t in range(T):
        total_outcomes = np.sum(CountOutcomeT[t])
        if total_outcomes > 0:
            percentages[t] = CountOutcomeT[t] / total_outcomes * 100

    
    outcome_labels = ['(C, C)', '(C, D)', '(D, C)', '(D, D)']
    plt.figure(figsize=(10, 6))

    for i in range(4):
        plt.plot(percentages[:, i], label=outcome_labels[i])

    plt.xlabel('Time Step')
    plt.ylabel('Percentage of Outcome')
    plt.title("Percentage of Prisoner's Dilemma Outcomes Over Time")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def softmax(x):
    z = x - np.max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax

if __name__ == '__main__':
    simStart = 1
    Nsim = 20
 
    tStart = 0
    T = 1000

    lr = 0.05

    algoName = 'Qpast1-b'

    Nact = 2
    gameName = 'PD2'

    Nagent = 20
    roundsG = 20

    mainTrain(gameName, Nact, roundsG, algoName, lr, Nagent, simStart, Nsim, tStart, T)




