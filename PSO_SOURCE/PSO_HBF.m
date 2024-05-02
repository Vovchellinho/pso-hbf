function z=Function(x)

%Sphere
z=sum((x-0.345).^2,2);

end

function [pop]=population(m,D,min_var,max_var)
pop=rand(m,D);
for i=1:m
pop(i,:)=pop(i,:).*(max_var(1,:)-min_var(1,:))+min_var(1,:);
end

                         
n_swarm=6;     %Number of initial population

n_var=1000;    %Number of initial Decision variables

min_var = -100.*ones(1,n_var);   % Lower Bound of Decision Variables
max_var = 100.*ones(1,n_var);    % Upper Bound of Decision Variables

w=.7298;
c2=1.4961;
c1=1.4961;

it1=0;
T_h=100;
EFs=0;
i=1;
%%%%%% Generating the initial population %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:n_swarm
    [pop(i,:)]=population(1,n_var,min_var,max_var);
    [Ccost(i)]=Function(pop(i,:));
    pop_cost(i,:)=[pop(i,:),Ccost(i)];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pop_cost1=pop_cost;

pop_cost_sort=sortrows(pop_cost,n_var+1);

g_best=pop_cost_sort(1,:);

p_best(1:n_swarm,:)=pop_cost(1:n_swarm,:);

v(:,:)=zeros(n_swarm,n_var);

upper_v=.1.*max_var;
lower_v=.1.*min_var;

%%%%%% Optimization loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while EFs<=3e6
    i=i+1;
    it=i-it1*T_h;
    if mod(it,T_h)==0   %% First part of HBF strategy
        loc=randperm(n_var);
        loc1=loc(1:round(n_var*rand-1)+1);
        Xs=randperm(n_swarm);
        pop(Xs(1),loc1)=rand(1,length(loc1)).*(max_var(loc1)-min_var(loc1))+min_var(loc1);
        [Ccost(1)]= Function(pop(Xs(1),1:n_var));
        EFs=EFs+1;
        p_best(Xs(1),:)=[pop(Xs(1),:),Ccost(1)];
        it1=it1+1;
    end

    two_pop=randperm(n_swarm);
    for j=two_pop(1:2)
        v(j,:)=c1*rand*(p_best(j,1:n_var)-pop_cost(j,1:n_var))+c2*rand*(g_best(1,1:n_var)-pop_cost(j,1:n_var))+w*v(j,1:n_var);

        loc=find(v(j,:)-upper_v>0);
        v(j,loc)=upper_v(loc).*rand(1,length(loc));
        loc=find(v(j,:)-lower_v<0);
        v(j,loc)=lower_v(loc).*rand(1,length(loc));
        v(j,:)=v(j,:).*round(rand(1,n_var)>0.8);

        pop(j,1:n_var)=pop_cost(j,1:n_var)+v(j,1:n_var);

        loc=find(pop(j,1:n_var)-max_var>0);
        pop(j,loc)=rand(1,length(loc)).*(max_var(loc)-min_var(loc))+min_var(loc);
        loc=find(min_var-pop(j,1:n_var)>0);
        pop(j,loc)=rand(1,length(loc)).*(max_var(loc)-min_var(loc))+min_var(loc);

        CCcost(j)=Function(pop(j,1:n_var));

        if CCcost(j)<=pop_cost(j,end)
            pop_cost1(j,:)=[pop(j,1:n_var),CCcost(j)];
        else
            pop_cost1(j,:)=pop_cost(j,:);
        end

        EFs=EFs+1;
%%%%%%%%%%%%%%%%%%%%%%%% Main part of HBF strategy %%%%%%%%%%%%%%%%%%%%%%%%
        vx=randn(1,n_var);
        beta=0.5+rand;
        stepsize=.00001*(it/T_h)*1.*((1./abs(vx)).^(1/beta)).*sign(vx);
        Xs=p_best(j,1:n_var)+(stepsize).*exp((it/10));
        Xa=p_best(j,1:n_var)+1*(stepsize).*(1-it/T_h).*(max_var-min_var);

        if j<n_swarm
            x=g_best(1,1:n_var);   %%%Select g_best(1,1:n_param) or p_best(j,1:n_param) or select between population and these vectors randomly
        else
            x=pop(j,1:n_var);
        end

        A=0;

        h0=exp(2-2*it/T_h);
        H=abs(2*rand(1,numel(x))*h0-h0);
        rr=rand(1,numel(x));
        cc=rand(1,numel(x));
        xx=find(rr<=A+cc*it/T_h) ;
        k1=zeros(1,n_var);
        k2=zeros(1,n_var);

        %  This section can be implemented in a loop for each decision variable %%%

        % for i1=1:numel(xx)
        % if H>.3*rand
        % k1(xx(i1))=1;
        % else
        % k2(xx(i1))=1;
        % end
        % end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if H>.3*rand
            k1(xx)=1;
        else
            k2(xx)=1;
        end

        xx=find(rr>A+cc*it/T_h) ;% hunting strategies condition
        k3=zeros(1,n_var);
        k3(xx)=1;
        z=Xs.*k1+Xa.*k2+k3.*x;

        loc=find(z-max_var>0);
        z(loc)=max_var(loc);
        loc=find(z-min_var<0);
        z(loc)=min_var(loc);

        zcost=Function(z);
        EFs=EFs+1;

        if zcost<  p_best(j,n_var+1)
            p_best(j,:)=[z,zcost]; % Can be replaced in p_best or pop
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if pop_cost1(j,end)<p_best(j,end)
            p_best(j,:)=pop_cost1(j,:);
        end
    end
    pop_cost=pop_cost1;
    pop_cost_sort=sortrows(pop_cost,n_var+1);
    pop_cost_sort1=sortrows(p_best,n_var+1);

    g_best1=pop_cost_sort1(1,:);

    if g_best1(1,end)<g_best(1,end)
        g_best=g_best1;
    end

    if mod(i,100)==0
        disp(['FEs>>   ' num2str(EFs) ': Best Cost = ' num2str(g_best(1,end)) ]);
    end

end
