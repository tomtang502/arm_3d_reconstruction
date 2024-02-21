import matplotlib.pyplot as plt

def show_3d_traj(inter_points_list, all_traj):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(inter_points_list)):
        inter_points=inter_points_list[i]
        ax.scatter(inter_points[:,0].detach().cpu(),
                inter_points[:,1].detach().cpu(),
                inter_points[:,2].detach().cpu(),s=1,c='g',alpha=0.05)

        # ax.scatter(xyz_combined[:,0].detach().cpu(),
        #            xyz_combined[:,1].detach().cpu(),
        #            xyz_combined[:,2].detach().cpu(),s=10,c='k')

        mean_inter_p=inter_points.mean(axis=0)

        ax.scatter(mean_inter_p[0].detach().cpu(),
                mean_inter_p[1].detach().cpu(),
                mean_inter_p[2].detach().cpu(),s=80,marker='X',c='r')


    ax.scatter(all_traj[:,0],all_traj[:,1],all_traj[:,2],s=5)

    ax.view_init(elev=70, azim=-0)
    ax.set_xlim(-0.4,0.4)
    ax.set_ylim(-0.4,0.4)
    ax.set_zlim(-0.4,0.4)
    fig.show()